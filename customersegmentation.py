import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# -------------------------------
# Generate Random Dataset
# -------------------------------
np.random.seed(42)
data = {
    'CustomerID': np.arange(1, 101),
    'Age': np.random.randint(18, 65, size=100),
    'Average_Spend': np.random.uniform(5, 50, size=100),
    'Visits_per_Week': np.random.uniform(1, 7, size=100),
    'Promotion_Interest': np.random.randint(1, 11, size=100)
}

df = pd.DataFrame(data)

# Features
X = df[['Age', 'Average_Spend', 'Visits_per_Week', 'Promotion_Interest']]

# -------------------------------
# Sidebar
# -------------------------------
st.title("Customer Segmentation App")
st.sidebar.markdown("### K-Means Clustering Project")

# -------------------------------
# Elbow Method for Optimal K
# -------------------------------
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(X)
    sse.append(kmeans_temp.inertia_)

st.header("📊 Elbow Method for Optimal K")
fig1, ax1 = plt.subplots()
ax1.plot(k_range, sse, marker='o')
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("SSE")
ax1.set_title("Elbow Method")
st.pyplot(fig1)

# -------------------------------
# Train Final KMeans Model
# -------------------------------
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Silhouette Score
sil_score = silhouette_score(X, df['Cluster'])

st.subheader(f"✅ Silhouette Score: **{sil_score:.2f}**")

# -------------------------------
# Scatter Plot (Age vs Spend)
# -------------------------------
st.header("📍 Customer Clusters (Age vs Average Spend)")

fig2, ax2 = plt.subplots()
scatter = ax2.scatter(df['Age'], df['Average_Spend'], c=df['Cluster'], cmap='viridis', s=100, alpha=0.7)
ax2.set_xlabel("Age")
ax2.set_ylabel("Average Spend")
ax2.set_title("Customer Clustering Visualization")
plt.colorbar(scatter, ax=ax2, label='Cluster')
st.pyplot(fig2)

# -------------------------------
# Prediction Function
# -------------------------------
def clustering(age, avg_spend, visit_per_week, promotion_interest):
    new_customer = np.array([[age, avg_spend, visit_per_week, promotion_interest]])
    predicted_cluster = kmeans.predict(new_customer)

    if predicted_cluster[0] == 0:
        return "🟢 Daily Shopper"
    elif predicted_cluster[0] == 1:
        return "🟡 Weekend Shopper"
    else:
        return "🔵 Promotion-Hunting Shopper"

# -------------------------------
# User Input Section
# -------------------------------
st.header("🔮 Predict New Customer Segment")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
avg_spend = st.number_input("Average Spend ($)", min_value=1.0, max_value=100.0, value=20.0)
visit = st.number_input("Visits per Week", min_value=0.0, max_value=14.0, value=3.0)
promo = st.slider("Promotion Interest (1 to 10)", 0, 10, 5)

if st.button("Predict Cluster"):
    cluster_result = clustering(age, avg_spend, visit, promo)
    st.success(f"### 🎯 Customer belongs to: **{cluster_result}**")

# -------------------------------
# Download Dataset
# -------------------------------
st.header("📥 Download Generated Dataset")

csv_data = df.to_csv(index=False)
st.download_button(
    label="Download CSV File",
    data=csv_data,
    file_name="customer_clusters.csv",
    mime="text/csv"
)
