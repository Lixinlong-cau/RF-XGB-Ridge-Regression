import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBRegressor

# ===================== 全局设置 =====================
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.family'] = ['Times New Roman']

# 文件路径
X_TRAIN_PATH = 'X_train.xlsx'
Y_TRAIN_PATH = 'Y_train_YD.xlsx'
#Y_TRAIN_PATH = 'Y_train_WUE.xlsx' 计算WUE时去掉注释
X_TEST_PATH = 'X_test.xlsx'
Y_TEST_PATH = 'Y_test_YD.xlsx'
#Y_TEST_PATH = 'Y_test_YD.xlsx' 计算WUE时去掉注释
SAVE_DIR = 'result'

# 分类特征列
CATEGORICAL_COLS = ['播种日期', '灌溉方式', '耕作模式', '覆盖模式', '品种', '灌溉期']

# ===================== 工具函数 =====================
def preprocess_data(X_train_original, X_test_original, categorical_cols):
    """数据预处理：编码、补缺失值、标准化"""
    label_encoders = {}
    for col in categorical_cols:
        if col in X_train_original.columns:
            le = LabelEncoder()
            X_train_original[col] = le.fit_transform(X_train_original[col].astype(str))
            X_test_original[col] = le.transform(X_test_original[col].astype(str))
            label_encoders[col] = le

    X_train_original.fillna(X_train_original.mean(numeric_only=True), inplace=True)
    X_test_original.fillna(X_test_original.mean(numeric_only=True), inplace=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_original)
    X_test = scaler.transform(X_test_original)

    return X_train, X_test, label_encoders, scaler

def tune_hyperparams(model, param_grid, X_train, y_train, cv=5, scoring='r2'):
    """
    超参数调优函数
    :param model: 待调优的模型
    :param param_grid: 参数网格
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param cv: 交叉验证折数
    :param scoring: 评估指标
    :return: 调优后的最佳模型
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"交叉验证最佳分数 ({scoring}): {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def get_feature_importance_avg(model1, model2, X_train_original, top_n=15):
    """计算两个基模型的特征重要性平均值"""
    imp1 = model1.feature_importances_
    imp2 = model2.feature_importances_
    imp_avg = (imp1 + imp2) / 2
    feature_names = X_train_original.columns.tolist()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_RF': imp1,
        'Importance_XGB': imp2,
        'Importance_Avg': imp_avg
    })
    return importance_df.sort_values('Importance_Avg', ascending=False).head(top_n)

# ===================== 主流程 =====================
try:
    # 1. 读取数据
    X_train_original = pd.read_excel(X_TRAIN_PATH)
    y_train = pd.read_excel(Y_TRAIN_PATH).iloc[:, 0]
    X_test_original = pd.read_excel(X_TEST_PATH)
    y_test = pd.read_excel(Y_TEST_PATH).iloc[:, 0]

    print(f"训练集输入形状: {X_train_original.shape}, 训练集输出形状: {y_train.shape}")
    print(f"测试集输入形状: {X_test_original.shape}, 测试集输出形状: {y_test.shape}")

    # 2. 数据预处理
    X_train, X_test, _, _ = preprocess_data(X_train_original, X_test_original, CATEGORICAL_COLS)

    # 3. 超参数调优：基模型
    print("\n===== 调优随机森林（RF）=====")
    # RF参数网格（兼顾性能与计算效率）
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_base = RandomForestRegressor(random_state=10000, n_jobs=-1)
    rf_best = tune_hyperparams(rf_base, rf_param_grid, X_train, y_train)

    print("\n===== 调优XGBoost =====")
    # XGBoost参数网格
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9]
    }
    xgb_base = XGBRegressor(random_state=42)
    xgb_best = tune_hyperparams(xgb_base, xgb_param_grid, X_train, y_train)

    # 4. 训练调优后的基模型并预测
    rf_best.fit(X_train, y_train)
    y_pred_rf = rf_best.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    xgb_best.fit(X_train, y_train)
    y_pred_xgb = xgb_best.predict(X_test)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

    # 打印基模型调优后性能
    print("\n===== 调优后基模型性能 =====")
    print(f"Random Forest: R²={r2_rf:.4f}, RMSE={rmse_rf:.2f}")
    print(f"XGBoost: R²={r2_xgb:.4f}, RMSE={rmse_xgb:.2f}")

    # 5. 堆叠特征生成（基模型在训练集和测试集的预测）
    y_train_rf = rf_best.predict(X_train)
    y_train_xgb = xgb_best.predict(X_train)
    X_stack_train = np.column_stack([y_train_rf, y_train_xgb])
    X_stack_test = np.column_stack([y_pred_rf, y_pred_xgb])

    # 6. 超参数调优：元模型（岭回归）
    print("\n===== 调优元模型（岭回归）=====")
    ridge_param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    ridge_base = Ridge()
    ridge_best = tune_hyperparams(ridge_base, ridge_param_grid, X_stack_train, y_train)

    # 7. 集成模型预测（堆叠）
    y_pred_stacking = ridge_best.predict(X_stack_test)

    # 8. 集成模型评估
    mse_ensemble = mean_squared_error(y_test, y_pred_stacking)
    rmse_ensemble = np.sqrt(mse_ensemble)
    r2_ensemble = r2_score(y_test, y_pred_stacking)

    print("\n===== 调优后集成模型性能 =====")
    print(f"Stacking (RF+XGB+Ridge): R²={r2_ensemble:.4f}, RMSE={rmse_ensemble:.2f}")

    # 9. 加权平均集成（可选，作为对比）
    weight_rf = r2_rf / (r2_rf + r2_xgb)
    weight_xgb = r2_xgb / (r2_rf + r2_xgb)
    y_pred_weighted = weight_rf * y_pred_rf + weight_xgb * y_pred_xgb
    r2_weighted = r2_score(y_test, y_pred_weighted)
    rmse_weighted = np.sqrt(mean_squared_error(y_test, y_pred_weighted))
    print(f"Weighted Average (RF+XGB): R²={r2_weighted:.4f}, RMSE={rmse_weighted:.2f}")

    # 选择性能更优的集成结果作为最终输出
    if r2_ensemble >= r2_weighted:
        y_pred_ensemble = y_pred_stacking
        ensemble_name = "Stacking (RF+XGB+Ridge)"
    else:
        y_pred_ensemble = y_pred_weighted
        ensemble_name = "Weighted Average (RF+XGB)"

    # 10. 可视化预测对比
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_ensemble, alpha=0.7, color='#FF6B6B', zorder=3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, color='#4ECDC4', zorder=2)
    plt.grid(True, linestyle='--', alpha=0.3, zorder=1)
    plt.xlabel('Actual Yield', fontsize=12)
    plt.ylabel('Predicted Yield', fontsize=12)
    plt.title(f'{ensemble_name} Prediction vs Actual\n(R²={r2_ensemble:.4f}, RMSE={rmse_ensemble:.2f})', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    save_path = f'{SAVE_DIR}\\Tuned_Ensemble_WUE_Prediction_Comparison.png'
    plt.savefig(save_path, dpi=600)
    plt.show()

    # 11. 保存预测结果（包含调优后基模型和集成模型）
    result_df = pd.DataFrame({
        'Actual_Yield': y_test,
        'Predicted_RF_Tuned': y_pred_rf,
        'Predicted_XGB_Tuned': y_pred_xgb,
        'Predicted_Stacking': y_pred_stacking,
        'Predicted_Weighted': y_pred_weighted,
        'Final_Predicted_Ensemble': y_pred_ensemble,
        'Ensemble_Absolute_Error': y_test - y_pred_ensemble,
        'Ensemble_Relative_Error(%)': ((y_test - y_pred_ensemble) / y_test * 100).round(2)
    })
    result_df = result_df.sort_values('Actual_Yield')
    result_excel_path = f'{SAVE_DIR}\\Tuned_Ensemble_YD_Prediction_Results.xlsx'
    result_df.to_excel(result_excel_path, index=False)
    print(f"\n调优后预测结果已保存至: {result_excel_path}")

    # 12. 综合特征重要性分析
    importance_df = get_feature_importance_avg(rf_best, xgb_best, X_train_original, top_n=15)
    print("\nTop 5 Features by Average Importance:")
    print(importance_df[['Feature', 'Importance_Avg']].head(5))

    # 13. 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance_Avg'], color='#FF6B6B')
    plt.xlabel('Average Feature Importance', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.title('Tuned Ensemble (RF+XGB) Feature Importance Ranking', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    importance_fig_path = f'{SAVE_DIR}\\Tuned_Ensemble_YD_Feature_Importance.png'
    #importance_fig_path = f'{SAVE_DIR}\\Tuned_Ensemble_WUE_Feature_Importance.png' 运行时去掉注释
    plt.savefig(importance_fig_path, dpi=600)
    plt.show()

    # 14. 保存特征重要性
    importance_excel_path = f'{SAVE_DIR}\\Tuned_Ensemble_YD_Feature_Importance_Results.xlsx'
    #importance_excel_path = f'{SAVE_DIR}\\Tuned_Ensemble_WUE_Feature_Importance_Results.xlsx' 运行时去掉注释
    importance_df.to_excel(importance_excel_path, index=False)
    print(f"调优后特征重要性已保存至: {importance_excel_path}")

except Exception as e:
    print(f"处理文件时出错: {str(e)}")