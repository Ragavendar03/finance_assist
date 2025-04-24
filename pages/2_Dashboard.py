import streamlit as st
import pandas as pd

st.set_page_config(page_title="📊 Expense Dashboard", layout="wide")
st.title("📊 Expense Analytics Dashboard")

# Ensure CSV is uploaded
if 'expense_df' not in st.session_state:
    st.warning("⚠️ No CSV uploaded. Please upload one from the Home page.")
    st.stop()

df = st.session_state['expense_df']

# Convert date column to datetime if not already
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Preview data
st.subheader("📄 Data Preview")
st.dataframe(df.head(), use_container_width=True)

st.markdown("---")

# Optional date filtering
st.subheader("📅 Filter by Date Range")
min_date, max_date = df['date'].min(), df['date'].max()
start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])

filtered_df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

st.markdown("---")

# Aggregation
st.subheader("📊 Summary & Category Breakdown")

# Convert amount if needed
if filtered_df['amount'].dtype not in ['float64', 'int64']:
    filtered_df['amount'] = pd.to_numeric(filtered_df['amount'], errors='coerce').fillna(0)

category_data = filtered_df.groupby('category')['amount'].sum().sort_values(ascending=False)

# Metrics
total_spend = category_data.sum()
top_category = category_data.idxmax()
top_amount = category_data.max()

m1, m2, m3 = st.columns(3)
m1.metric("Total Spend", f"₹{total_spend:,.2f}")
m2.metric("Top Category", top_category)
m3.metric("Top Spend", f"₹{top_amount:,.2f}")

st.markdown("---")

# Chart
st.subheader("📈 Spend by Category")
chart_type = st.radio("Chart Type", ["Bar Chart", "Line Chart", "Area Chart"], horizontal=True)

chart_df = category_data.reset_index()
chart_df.columns = ['category', 'amount']

if chart_type == "Bar Chart":
    st.bar_chart(chart_df.set_index('category'))
elif chart_type == "Line Chart":
    st.line_chart(chart_df.set_index('category'))
elif chart_type == "Area Chart":
    st.area_chart(chart_df.set_index('category'))

# Account-based breakdown
st.markdown("---")
st.subheader("🏦 Account Breakdown")
account_data = filtered_df.groupby('account_name')['amount'].sum().sort_values(ascending=False)
st.bar_chart(account_data)

# st.markdown("---")
# if st.button("🏠 Back to Home"):
#     st.switch_page("app.py")
