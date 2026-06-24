import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. THE FOOLPROOF FILE PATH FIX
# ==========================================
# This forces Python to look exactly in the folder where this script is saved
current_folder = os.path.dirname(os.path.abspath(__file__))
jpn_path = os.path.join(current_folder, 'dataset', 'JPN Data.xlsx')
ind_path = os.path.join(current_folder, 'dataset', 'IN_Data.xlsx')

# load data
print("Finding files...")
df_jpn = pd.read_excel(jpn_path)
df_ind = pd.read_excel(ind_path)
print("Files loaded successfully!")

# The exact column name in your Japanese dataset
target = 'PURCHASE'

# func to get segment
def get_seg(age):
    if pd.isna(age): return np.nan
    if age < 200: return 1
    elif age <= 360: return 2
    elif age <= 500: return 3
    else: return 4

# apply to japan (using exact UPPERCASE column names)
df_jpn['SEGMENT'] = df_jpn['AGE_CAR'].apply(get_seg)

# prep india data and fix dates
curr_date = pd.to_datetime('2019-07-01')
df_ind['DT_MAINT'] = pd.to_datetime(df_ind['DT_MAINT'])

# Calculate the new AGE_CAR column for India
df_ind['AGE_CAR'] = (curr_date - df_ind['DT_MAINT']).dt.days
df_ind['SEGMENT'] = df_ind['AGE_CAR'].apply(get_seg)

# convert rupees to yen (1.55 rate)
df_ind['ANN_INCOME_JPY'] = df_ind['ANN_INCOME'] * 1.55

# setup model features exactly as they are capitalized
features = ['ANN_INCOME', 'AGE_CAR', 'SEGMENT']

# drop missing so it doesnt crash
train = df_jpn.dropna(subset=features + [target])

# ==========================================
# REQUIREMENT 4: MODEL EVALUATION METRICS
# ==========================================
print("\n" + "="*40)
print("MODEL EVALUATION METRICS (For Report)")
print("="*40)

# Split data 80% for training, 20% for testing
X_eval = train[features]
y_eval = train[target]
X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42)

# Train a temporary model just for testing
test_model = LogisticRegression(max_iter=1000)
test_model.fit(X_train_eval, y_train_eval)

# Check accuracy
y_pred_eval = test_model.predict(X_test_eval)
accuracy = accuracy_score(y_test_eval, y_pred_eval)

print(f"Accuracy Score: {accuracy * 100:.2f}%\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test_eval, y_pred_eval))
print("\nDetailed Classification Report:")
print(classification_report(y_test_eval, y_pred_eval))


# ==========================================
# FINAL MODEL & INDIA PREDICTIONS
# ==========================================
model = LogisticRegression(max_iter=1000)
model.fit(train[features], train[target])

X_ind = df_ind[['ANN_INCOME_JPY', 'AGE_CAR', 'SEGMENT']].rename(columns={'ANN_INCOME_JPY': 'ANN_INCOME'})
X_ind = X_ind.fillna(X_ind.median()) 

df_ind['PREDICTED_PURCHASE'] = model.predict(X_ind)
total_customers = df_ind['PREDICTED_PURCHASE'].sum()

print("\n" + "="*50)
print(f"TOTAL PROJECTED INDIA CUSTOMERS: {total_customers}")
print("-" * 50)

if total_customers >= 12000:
    print("✅ BUSINESS GOAL MET: Forecast exceeds 12,000 minimum.")
else:
    print("❌ BUSINESS GOAL NOT MET: Forecast below 12,000 minimum.")
print("="*50)

b0, b1, b2, b3 = model.intercept_[0], model.coef_[0][0], model.coef_[0][1], model.coef_[0][2]
print(f"\nEquation: z = {b0:.6f} + ({b1:.6f} * Income) + ({b2:.6f} * Age) + ({b3:.6f} * Segment)")

# ==========================================
# DATA VISUALIZATIONS (DASHBOARD)
# ==========================================
print("\nGenerating visual dashboard...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('ABG Motors: India Expansion Forecast Dashboard', fontsize=20, fontweight='bold')

axes[0].bar(['Projected Indian Sales'], [total_customers], color='#2ecc71')
axes[0].axhline(y=12000, color='#e74c3c', linestyle='--', linewidth=3, label='Minimum Goal (12,000)')
axes[0].set_title('Forecast vs. Business Goal', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].text(0, total_customers/2, f"{total_customers:,}", ha='center', va='center', color='white', fontsize=18, fontweight='bold')

segment_sales = df_ind[df_ind['PREDICTED_PURCHASE'] == 1].groupby('SEGMENT').size().reset_index(name='Sales')
sns.barplot(data=segment_sales, x='SEGMENT', y='Sales', ax=axes[1], palette='Blues')
axes[1].set_title('Projected Sales by Customer Segment', fontsize=14, fontweight='bold')

sns.barplot(data=df_ind, x='PREDICTED_PURCHASE', y='ANN_INCOME_JPY', ax=axes[2], palette='Set2', estimator=np.mean)
axes[2].set_title('Average Income: Buyers vs. Non-Buyers', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 

# ==========================================
# SAVING FILES TO CORRECT FOLDERS
# ==========================================
# 1. Create the presentation folder if it doesn't exist
presentation_folder = os.path.join(current_folder, 'presentation')
os.makedirs(presentation_folder, exist_ok=True)

# 2. Define the new save paths
save_img = os.path.join(presentation_folder, 'ABG_Motors_Dashboard.png')
save_jpn = os.path.join(current_folder, 'dataset', 'JPN_Data_Modified.xlsx')
save_ind = os.path.join(current_folder, 'dataset', 'IN_Data_Predicted.xlsx')

# 3. Save the files
plt.savefig(save_img, dpi=300, bbox_inches='tight')
df_jpn.to_excel(save_jpn, index=False)
df_ind.to_excel(save_ind, index=False)

print("Success! Excel files saved to 'dataset'. Dashboard image saved to 'presentation' folder.")