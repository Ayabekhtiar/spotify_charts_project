"""Machine learning model training and evaluation functions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def train_stream_prediction_models(X_train, y_train, X_test, y_test, available_features):
    """Train regression models for stream prediction and return results."""
    # Check XGBoost availability
    xgboost_available = XGBOOST_AVAILABLE
    if not xgboost_available:
        print("XGBoost not available. Install with: pip install xgboost")
    
    # Scale features for linear regression
    scaler_reg = StandardScaler()
    X_train_scaled = scaler_reg.fit_transform(X_train)
    X_test_scaled = scaler_reg.transform(X_test)
    
    print("=" * 60)
    print("STREAM PREDICTION MODELS")
    print("=" * 60)
    
    models = {}
    predictions = {}
    results = {}
    
    # Model 1: Linear Regression (baseline)
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_mae = mean_absolute_error(y_test, y_pred_lr)
    lr_r2 = r2_score(y_test, y_pred_lr)
    
    models['lr'] = lr_model
    predictions['lr'] = y_pred_lr
    results['lr'] = {'rmse': lr_rmse, 'mae': lr_mae, 'r2': lr_r2}
    
    print(f"\n1. Linear Regression:")
    print(f"   RMSE: {lr_rmse:,.0f}")
    print(f"   MAE: {lr_mae:,.0f}")
    print(f"   R² Score: {lr_r2:.4f}")
    
    # Model 2: Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    models['rf'] = rf_model
    predictions['rf'] = y_pred_rf
    results['rf'] = {'rmse': rf_rmse, 'mae': rf_mae, 'r2': rf_r2}
    
    print(f"\n2. Random Forest Regressor:")
    print(f"   RMSE: {rf_rmse:,.0f}")
    print(f"   MAE: {rf_mae:,.0f}")
    print(f"   R² Score: {rf_r2:.4f}")
    
    # Model 3: XGBoost (if available)
    if xgboost_available:
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        
        xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
        xgb_r2 = r2_score(y_test, y_pred_xgb)
        
        models['xgb'] = xgb_model
        predictions['xgb'] = y_pred_xgb
        results['xgb'] = {'rmse': xgb_rmse, 'mae': xgb_mae, 'r2': xgb_r2}
        
        print(f"\n3. XGBoost Regressor:")
        print(f"   RMSE: {xgb_rmse:,.0f}")
        print(f"   MAE: {xgb_mae:,.0f}")
        print(f"   R² Score: {xgb_r2:.4f}")
    else:
        models['xgb'] = None
        predictions['xgb'] = None
        results['xgb'] = None
    
    return {
        'models': models,
        'predictions': predictions,
        'results': results,
        'xgboost_available': xgboost_available,
        'scaler': scaler_reg
    }


def plot_prediction_results(y_test, predictions_dict, model_names):
    """Plot predicted vs actual for regression models."""
    xgboost_available = predictions_dict.get('xgb') is not None
    
    fig, axes = plt.subplots(1, 3 if xgboost_available else 2, figsize=(18, 5))
    if not xgboost_available:
        axes = [axes[0], axes[1]]
    
    # Sample for visualization (to avoid overcrowding)
    sample_idx = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
    
    # Linear Regression
    if 'lr' in predictions_dict and predictions_dict['lr'] is not None:
        axes[0].scatter(y_test.iloc[sample_idx], predictions_dict['lr'][sample_idx], alpha=0.3, s=10)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Streams', fontsize=11)
        axes[0].set_ylabel('Predicted Streams', fontsize=11)
        axes[0].set_title(f'Linear Regression', fontsize=12)
        axes[0].grid(True, alpha=0.3)
    
    # Random Forest
    if 'rf' in predictions_dict and predictions_dict['rf'] is not None:
        axes[1].scatter(y_test.iloc[sample_idx], predictions_dict['rf'][sample_idx], alpha=0.3, s=10, color='green')
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Streams', fontsize=11)
        axes[1].set_ylabel('Predicted Streams', fontsize=11)
        axes[1].set_title(f'Random Forest', fontsize=12)
        axes[1].grid(True, alpha=0.3)
    
    # XGBoost
    if xgboost_available and 'xgb' in predictions_dict:
        axes[2].scatter(y_test.iloc[sample_idx], predictions_dict['xgb'][sample_idx], alpha=0.3, s=10, color='purple')
        axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[2].set_xlabel('Actual Streams', fontsize=11)
        axes[2].set_ylabel('Predicted Streams', fontsize=11)
        axes[2].set_title(f'XGBoost', fontsize=12)
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Stream Prediction: Predicted vs Actual', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def prepare_classification_data(ml_data, available_features, train_weeks, test_weeks):
    """Prepare classification dataset with positive and negative samples."""
    # Get all unique track_ids and week_dates
    all_tracks = ml_data['track_id'].unique()
    all_weeks = sorted(ml_data['week_date'].unique())
    
    # Create a complete grid of track-week combinations that actually appeared
    appeared_combinations = set(zip(ml_data['track_id'], ml_data['week_date']))
    
    # Sample negative examples (tracks that could appear but didn't)
    # For each week, sample some tracks that didn't appear
    negative_samples = []
    for week in all_weeks:
        week_tracks = set(ml_data[ml_data['week_date'] == week]['track_id'].unique())
        # Sample tracks that appeared in other weeks but not this week
        potential_tracks = set(all_tracks) - week_tracks
        # Sample up to the same number as positive samples for balance
        n_negative = min(len(week_tracks), len(potential_tracks))
        sampled_negative = np.random.choice(list(potential_tracks), n_negative, replace=False)
        for track_id in sampled_negative:
            negative_samples.append({'track_id': track_id, 'week_date': week, 'appeared': 0})
    
    # Create positive samples
    positive_samples = ml_data[['track_id', 'week_date']].copy()
    positive_samples['appeared'] = 1
    
    # Combine and merge with features
    classification_data = pd.concat([
        positive_samples,
        pd.DataFrame(negative_samples)
    ], ignore_index=True)
    
    # Merge with features (for positive samples, use actual data; for negative, use track-level features)
    # For negative samples, we'll use the most recent available features for that track
    classification_data = classification_data.merge(
        ml_data[['track_id', 'week_date'] + available_features],
        on=['track_id', 'week_date'],
        how='left'
    )
    
    # For negative samples without features, use the latest available features for that track
    for idx, row in classification_data[classification_data['appeared'] == 0].iterrows():
        if pd.isna(row[available_features[0]]):
            track_features = ml_data[
                (ml_data['track_id'] == row['track_id']) & 
                (ml_data['week_date'] <= row['week_date'])
            ]
            if len(track_features) > 0:
                latest_features = track_features.sort_values('week_date').iloc[-1]
                for feat in available_features:
                    classification_data.at[idx, feat] = latest_features[feat]
    
    # Fill remaining NaN with median
    for feat in available_features:
        classification_data[feat] = classification_data[feat].fillna(classification_data[feat].median())
    
    # Remove rows that still have issues
    classification_data_clean = classification_data.dropna(subset=available_features + ['appeared'])
    
    # Time-based split for classification
    train_data_clf = classification_data_clean[classification_data_clean['week_date'].isin(train_weeks)]
    test_data_clf = classification_data_clean[classification_data_clean['week_date'].isin(test_weeks)]
    
    X_train_clf = train_data_clf[available_features]
    y_train_clf = train_data_clf['appeared']
    X_test_clf = test_data_clf[available_features]
    y_test_clf = test_data_clf['appeared']
    
    print(f"\nClassification dataset:")
    print(f"Training set: {len(X_train_clf)} samples ({y_train_clf.sum()} positive, {len(y_train_clf) - y_train_clf.sum()} negative)")
    print(f"Test set: {len(X_test_clf)} samples ({y_test_clf.sum()} positive, {len(y_test_clf) - y_test_clf.sum()} negative)")
    print(f"Class balance (train): {y_train_clf.mean():.2%} positive")
    
    return X_train_clf, y_train_clf, X_test_clf, y_test_clf


def train_classification_models(X_train, y_train, X_test, y_test, available_features):
    """Train classification models for appearance prediction and return results."""
    # Check XGBoost availability
    xgboost_available = XGBOOST_AVAILABLE
    
    # Scale features for logistic regression
    scaler_clf = StandardScaler()
    X_train_scaled = scaler_clf.fit_transform(X_train)
    X_test_scaled = scaler_clf.transform(X_test)
    
    print("=" * 60)
    print("APPEARANCE CLASSIFICATION MODELS")
    print("=" * 60)
    
    models = {}
    predictions = {}
    predictions_proba = {}
    results = {}
    
    # Model 1: Logistic Regression (baseline)
    lr_clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    lr_clf.fit(X_train_scaled, y_train)
    y_pred_lr_clf = lr_clf.predict(X_test_scaled)
    y_pred_proba_lr_clf = lr_clf.predict_proba(X_test_scaled)[:, 1]
    
    lr_clf_acc = accuracy_score(y_test, y_pred_lr_clf)
    lr_clf_prec = precision_score(y_test, y_pred_lr_clf)
    lr_clf_rec = recall_score(y_test, y_pred_lr_clf)
    lr_clf_f1 = f1_score(y_test, y_pred_lr_clf)
    lr_clf_auc = roc_auc_score(y_test, y_pred_proba_lr_clf)
    
    models['lr'] = lr_clf
    predictions['lr'] = y_pred_lr_clf
    predictions_proba['lr'] = y_pred_proba_lr_clf
    results['lr'] = {'acc': lr_clf_acc, 'prec': lr_clf_prec, 'rec': lr_clf_rec, 'f1': lr_clf_f1, 'auc': lr_clf_auc}
    
    print(f"\n1. Logistic Regression:")
    print(f"   Accuracy: {lr_clf_acc:.4f}")
    print(f"   Precision: {lr_clf_prec:.4f}")
    print(f"   Recall: {lr_clf_rec:.4f}")
    print(f"   F1-Score: {lr_clf_f1:.4f}")
    print(f"   ROC-AUC: {lr_clf_auc:.4f}")
    
    # Model 2: Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    y_pred_rf_clf = rf_clf.predict(X_test)
    y_pred_proba_rf_clf = rf_clf.predict_proba(X_test)[:, 1]
    
    rf_clf_acc = accuracy_score(y_test, y_pred_rf_clf)
    rf_clf_prec = precision_score(y_test, y_pred_rf_clf)
    rf_clf_rec = recall_score(y_test, y_pred_rf_clf)
    rf_clf_f1 = f1_score(y_test, y_pred_rf_clf)
    rf_clf_auc = roc_auc_score(y_test, y_pred_proba_rf_clf)
    
    models['rf'] = rf_clf
    predictions['rf'] = y_pred_rf_clf
    predictions_proba['rf'] = y_pred_proba_rf_clf
    results['rf'] = {'acc': rf_clf_acc, 'prec': rf_clf_prec, 'rec': rf_clf_rec, 'f1': rf_clf_f1, 'auc': rf_clf_auc}
    
    print(f"\n2. Random Forest Classifier:")
    print(f"   Accuracy: {rf_clf_acc:.4f}")
    print(f"   Precision: {rf_clf_prec:.4f}")
    print(f"   Recall: {rf_clf_rec:.4f}")
    print(f"   F1-Score: {rf_clf_f1:.4f}")
    print(f"   ROC-AUC: {rf_clf_auc:.4f}")
    
    # Model 3: XGBoost (if available)
    if xgboost_available:
        xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb_clf.fit(X_train, y_train)
        y_pred_xgb_clf = xgb_clf.predict(X_test)
        y_pred_proba_xgb_clf = xgb_clf.predict_proba(X_test)[:, 1]
        
        xgb_clf_acc = accuracy_score(y_test, y_pred_xgb_clf)
        xgb_clf_prec = precision_score(y_test, y_pred_xgb_clf)
        xgb_clf_rec = recall_score(y_test, y_pred_xgb_clf)
        xgb_clf_f1 = f1_score(y_test, y_pred_xgb_clf)
        xgb_clf_auc = roc_auc_score(y_test, y_pred_proba_xgb_clf)
        
        models['xgb'] = xgb_clf
        predictions['xgb'] = y_pred_xgb_clf
        predictions_proba['xgb'] = y_pred_proba_xgb_clf
        results['xgb'] = {'acc': xgb_clf_acc, 'prec': xgb_clf_prec, 'rec': xgb_clf_rec, 'f1': xgb_clf_f1, 'auc': xgb_clf_auc}
        
        print(f"\n3. XGBoost Classifier:")
        print(f"   Accuracy: {xgb_clf_acc:.4f}")
        print(f"   Precision: {xgb_clf_prec:.4f}")
        print(f"   Recall: {xgb_clf_rec:.4f}")
        print(f"   F1-Score: {xgb_clf_f1:.4f}")
        print(f"   ROC-AUC: {xgb_clf_auc:.4f}")
    else:
        models['xgb'] = None
        predictions['xgb'] = None
        predictions_proba['xgb'] = None
        results['xgb'] = None
    
    return {
        'models': models,
        'predictions': predictions,
        'predictions_proba': predictions_proba,
        'results': results,
        'xgboost_available': xgboost_available,
        'scaler': scaler_clf
    }


def plot_classification_results(y_test, predictions_dict, model_names):
    """Plot confusion matrices for classification models."""
    xgboost_available = predictions_dict.get('xgb') is not None
    
    fig, axes = plt.subplots(1, 3 if xgboost_available else 2, figsize=(18, 5))
    if not xgboost_available:
        axes = [axes[0], axes[1]]
    
    # Get results from model_names dict (should contain results)
    results_dict = model_names.get('results', {})
    
    # Logistic Regression
    if 'lr' in predictions_dict and predictions_dict['lr'] is not None:
        cm_lr = confusion_matrix(y_test, predictions_dict['lr'])
        lr_results = results_dict.get('lr', {})
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_xlabel('Predicted', fontsize=11)
        axes[0].set_ylabel('Actual', fontsize=11)
        acc = lr_results.get('acc', 0)
        f1 = lr_results.get('f1', 0)
        axes[0].set_title(f'Logistic Regression\n(Acc={acc:.3f}, F1={f1:.3f})', fontsize=12)
    
    # Random Forest
    if 'rf' in predictions_dict and predictions_dict['rf'] is not None:
        cm_rf = confusion_matrix(y_test, predictions_dict['rf'])
        rf_results = results_dict.get('rf', {})
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
        axes[1].set_xlabel('Predicted', fontsize=11)
        axes[1].set_ylabel('Actual', fontsize=11)
        acc = rf_results.get('acc', 0)
        f1 = rf_results.get('f1', 0)
        axes[1].set_title(f'Random Forest\n(Acc={acc:.3f}, F1={f1:.3f})', fontsize=12)
    
    # XGBoost
    if xgboost_available and 'xgb' in predictions_dict:
        cm_xgb = confusion_matrix(y_test, predictions_dict['xgb'])
        xgb_results = results_dict.get('xgb', {})
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Purples', ax=axes[2], cbar=False)
        axes[2].set_xlabel('Predicted', fontsize=11)
        axes[2].set_ylabel('Actual', fontsize=11)
        acc = xgb_results.get('acc', 0)
        f1 = xgb_results.get('f1', 0)
        axes[2].set_title(f'XGBoost\n(Acc={acc:.3f}, F1={f1:.3f})', fontsize=12)
    
    plt.suptitle('Confusion Matrices: Appearance Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def compare_models(regression_results, classification_results):
    """Print model comparison summary."""
    print("=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    # Regression comparison
    if regression_results and regression_results.get('results'):
        reg_results = regression_results['results']
        xgboost_available = regression_results.get('xgboost_available', False)
        
        print("\nSTREAM PREDICTION (Regression):")
        print(f"{'Model':<20} {'RMSE':<15} {'MAE':<15} {'R²':<10}")
        print("-" * 60)
        
        if reg_results.get('lr'):
            lr = reg_results['lr']
            print(f"{'Linear Regression':<20} {lr['rmse']:>12,.0f}  {lr['mae']:>12,.0f}  {lr['r2']:>8.4f}")
        
        if reg_results.get('rf'):
            rf = reg_results['rf']
            print(f"{'Random Forest':<20} {rf['rmse']:>12,.0f}  {rf['mae']:>12,.0f}  {rf['r2']:>8.4f}")
        
        if xgboost_available and reg_results.get('xgb'):
            xgb = reg_results['xgb']
            print(f"{'XGBoost':<20} {xgb['rmse']:>12,.0f}  {xgb['mae']:>12,.0f}  {xgb['r2']:>8.4f}")
    
    # Classification comparison
    if classification_results and classification_results.get('results'):
        clf_results = classification_results['results']
        xgboost_available = classification_results.get('xgboost_available', False)
        
        print("\nAPPEARANCE CLASSIFICATION:")
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'ROC-AUC':<10}")
        print("-" * 80)
        
        if clf_results.get('lr'):
            lr = clf_results['lr']
            print(f"{'Logistic Regression':<20} {lr['acc']:>10.4f}  {lr['prec']:>10.4f}  {lr['rec']:>10.4f}  {lr['f1']:>8.4f}  {lr['auc']:>8.4f}")
        
        if clf_results.get('rf'):
            rf = clf_results['rf']
            print(f"{'Random Forest':<20} {rf['acc']:>10.4f}  {rf['prec']:>10.4f}  {rf['rec']:>10.4f}  {rf['f1']:>8.4f}  {rf['auc']:>8.4f}")
        
        if xgboost_available and clf_results.get('xgb'):
            xgb = clf_results['xgb']
            print(f"{'XGBoost':<20} {xgb['acc']:>10.4f}  {xgb['prec']:>10.4f}  {xgb['rec']:>10.4f}  {xgb['f1']:>8.4f}  {xgb['auc']:>8.4f}")


def plot_feature_importance(models_dict, available_features, top_n=15):
    """Plot feature importance for Random Forest models."""
    regression_model = models_dict.get('regression', {}).get('models', {}).get('rf')
    classification_model = models_dict.get('classification', {}).get('models', {}).get('rf')
    
    if regression_model is None and classification_model is None:
        print("No Random Forest models available for feature importance analysis.")
        return
    
    fig, axes = plt.subplots(1, 2 if (regression_model and classification_model) else 1, figsize=(16, 8))
    if not (regression_model and classification_model):
        axes = [axes]
    
    # Regression feature importance
    if regression_model:
        rf_feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': regression_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        axes[0].barh(range(top_n), rf_feature_importance.head(top_n)['importance'], color='steelblue')
        axes[0].set_yticks(range(top_n))
        axes[0].set_yticklabels(rf_feature_importance.head(top_n)['feature'], fontsize=10)
        axes[0].set_xlabel('Importance', fontsize=12)
        axes[0].set_title('Top 15 Features: Stream Prediction (Random Forest)', fontsize=13, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        print("\nTop 10 Features for Stream Prediction:")
        print(rf_feature_importance.head(10)[['feature', 'importance']].to_string(index=False))
    
    # Classification feature importance
    if classification_model:
        rf_clf_feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': classification_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if regression_model:
            ax_idx = 1
        else:
            ax_idx = 0
        
        axes[ax_idx].barh(range(top_n), rf_clf_feature_importance.head(top_n)['importance'], color='coral')
        axes[ax_idx].set_yticks(range(top_n))
        axes[ax_idx].set_yticklabels(rf_clf_feature_importance.head(top_n)['feature'], fontsize=10)
        axes[ax_idx].set_xlabel('Importance', fontsize=12)
        axes[ax_idx].set_title('Top 15 Features: Appearance Classification (Random Forest)', fontsize=13, fontweight='bold')
        axes[ax_idx].invert_yaxis()
        axes[ax_idx].grid(True, alpha=0.3, axis='x')
        
        print("\nTop 10 Features for Appearance Classification:")
        print(rf_clf_feature_importance.head(10)[['feature', 'importance']].to_string(index=False))
    
    plt.tight_layout()
    plt.show()

