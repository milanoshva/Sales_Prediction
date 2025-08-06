import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import sys
import os
from ui.styles import set_custom_ui, get_plotly_template

# Set up logging to app.txt
logging.basicConfig(level=logging.INFO, filename='app.txt')
logger = logging.getLogger(__name__)

# Apply custom UI
set_custom_ui()

# Plotly theme
plotly_template = get_plotly_template()

# Debug mode
logger.info(f"Analytics page accessed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Get language and mode from session state
lang = st.session_state.get('language', 'ID')
mode = st.session_state.get('mode', 'Normal')

# Enhanced Header
if lang == "ID":
    st.markdown("""
    <h1>📊 Dashboard Analisis Penjualan UMKM Kuliner</h1>
    <p style="font-size: 0.9rem;">Lihat tren penjualan, produk terlaris, dan KPI untuk merencanakan bisnis Anda.</p>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <h1>📊 Culinary SME's Sales Analytics Dashboard</h1>
    <p style="font-size: 0.9rem;">View sales trends, top products, and KPIs to plan your business.</p>
    """, unsafe_allow_html=True)

# Add Advanced mode badge
if mode == 'Advanced':
    st.markdown('<span class="advanced-badge">{}</span>'.format("Mode Lanjutan" if lang == "ID" else "Advanced Mode"), unsafe_allow_html=True)

# Check if data is available
if 'df' not in st.session_state or st.session_state.df.empty:
    if lang == "ID":
        st.markdown("""
        <div class="info-box">
            <h3>Selamat Datang di Dashboard Penjualan UMKM!</h3>
            <p>Untuk memulai, silakan unggah file CSV transaksi Anda di halaman <a href="/Home" target="_self">Utama</a>.</p>
            <p>Data yang diunggah akan dianalisis untuk menampilkan tren penjualan, produk terlaris, dan performa kategori.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h3>Welcome to the UMKM Sales Dashboard!</h3>
            <p>To get started, please upload your transaction CSV file on the <a href="/Home" target="_self">Home</a> page.</p>
            <p>The uploaded data will be analyzed to display sales trends, top products, and category performance.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    df = st.session_state.df.copy()

    # --- START Filter Section inside an expander ---
    # Wrap the entire filter section in st.expander
    with st.expander("📅 Tampilkan Filter Analisis" if lang == "ID" else "📅 Show Analysis Filters", expanded=False):
        col_time, col_product, col_transaction = st.columns(3)

        with col_time:
            st.markdown("### ⏳ Periode" if lang == "ID" else "### ⏳ Period")
            if not pd.api.types.is_datetime64_any_dtype(df['waktu']):
                df['waktu'] = pd.to_datetime(df['waktu'], errors='coerce')
                df.dropna(subset=['waktu'], inplace=True)

            if 'date_range' not in st.session_state:
                st.session_state.date_range = [df['waktu'].min().date(), df['waktu'].max().date()]

            date_range_tuple = st.date_input("Pilih Periode" if lang == "ID" else "Select Period",
                                    value=st.session_state.date_range,
                                    min_value=df['waktu'].min().date(),
                                    max_value=df['waktu'].max().date(),
                                    key="date_range_input")

            if date_range_tuple and len(date_range_tuple) == 2:
                st.session_state.date_range = list(date_range_tuple)

            time_granularity_options = ["Harian", "Mingguan", "Bulanan", "Tahunan"]
            if 'time_granularity' not in st.session_state:
                st.session_state.time_granularity = time_granularity_options[2]

            time_granularity = st.radio("Granularitas Waktu" if lang == "ID" else "Time Granularity",
                                        time_granularity_options,
                                        index=time_granularity_options.index(st.session_state.time_granularity),
                                        horizontal=True,
                                        key="time_granularity_radio",
                                        help="Pilih granularitas waktu untuk analisis" if lang == "ID" else "Select time granularity for analysis")
            st.session_state.time_granularity = time_granularity

        with col_product:
            st.markdown("### 🛒 Produk" if lang == "ID" else "### 🛒 Product")
            
            selected_products = st.multiselect("Pilih Produk" if lang == "ID" else "Select Products",
                                              sorted(df['nama_produk'].unique()),
                                              default=st.session_state.get('selected_products', []),
                                              key="selected_products_multiselect")
            st.session_state.selected_products = selected_products
            
            if 'kategori_produk' in df.columns:
                categories = sorted(df['kategori_produk'].unique())
                include_categories = st.checkbox("Saring berdasarkan Kategori" if lang == "ID" else "Filter by Category", 
                                                 value=st.session_state.get('include_categories', False),
                                                 key="include_categories_checkbox")
                st.session_state.include_categories = include_categories
                
                selected_categories = st.radio("Pilih Kategori" if lang == "ID" else "Select Category",
                                              categories,
                                              index=categories.index(st.session_state.get('selected_categories', categories[0])) if st.session_state.get('selected_categories') in categories else 0,
                                              horizontal=True,
                                              key="selected_categories_radio",
                                              disabled=not include_categories,
                                              help="Pilih kategori produk untuk analisis" if lang == "ID" else "Select product category for analysis")
                st.session_state.selected_categories = selected_categories
            else:
                selected_categories = None
            
            selected_skus = None

        with col_transaction:
            st.markdown("### 💳 Transaksi" if lang == "ID" else "### 💳 Transaction")
            
            if 'metode_pembayaran' in df.columns:
                payment_methods = sorted(df['metode_pembayaran'].unique())
                enable_payment_filter = st.checkbox("Saring berdasarkan Pembayaran" if lang == "ID" else "Filter by Payment Method", 
                                                    value=st.session_state.get('enable_payment_filter', False),
                                                    key="enable_payment_filter_checkbox")
                st.session_state.enable_payment_filter = enable_payment_filter
                
                selected_payment_method = st.radio("Pilih Metode Pembayaran" if lang == "ID" else "Select Payment Method",
                                                  payment_methods,
                                                  index=payment_methods.index(st.session_state.get('selected_payment_method', payment_methods[0])) if st.session_state.get('selected_payment_method') in payment_methods else 0,
                                                  horizontal=True,
                                                  key="selected_payment_method_radio",
                                                  disabled=not enable_payment_filter,
                                                  help="Pilih metode pembayaran untuk analisis" if lang == "ID" else "Select payment method for analysis")
                st.session_state.selected_payment_method = selected_payment_method
            else:
                selected_payment_method = None
            
            if 'tipe_pesanan' in df.columns:
                order_types = sorted(df['tipe_pesanan'].unique())
                enable_order_type_filter = st.checkbox("Saring berdasarkan Tipe Pesanan" if lang == "ID" else "Filter by Order Type", 
                                                       value=st.session_state.get('enable_order_type_filter', False),
                                                       key="enable_order_type_filter_checkbox")
                st.session_state.enable_order_type_filter = enable_order_type_filter
                
                selected_order_type = st.radio("Pilih Tipe Pesanan" if lang == "ID" else "Select Order Type",
                                              order_types,
                                              index=order_types.index(st.session_state.get('selected_order_type', order_types[0])) if st.session_state.get('selected_order_type') in order_types else 0,
                                              horizontal=True,
                                              key="selected_order_type_radio",
                                              disabled=not enable_order_type_filter,
                                              help="Pilih tipe pesanan (Dine-in, Takeaway) untuk analisis" if lang == "ID" else "Select order type (Dine-in, Takeaway) for analysis")
                st.session_state.selected_order_type = selected_order_type
            else:
                selected_order_type = None

        # Reset Button inside the expander
        if st.button("Reset Semua Filter" if lang == "ID" else "Reset All Filters", key="reset_filters_button"):
            keys_to_reset = [
                'date_range', 'time_granularity', 'selected_products', 'include_categories', 
                'selected_categories', 'enable_payment_filter', 'selected_payment_method', 
                'enable_order_type_filter', 'selected_order_type'
            ]
            for key in keys_to_reset:
                st.session_state.pop(key, None)
            
            if lang == "ID":
                st.info("Filter direset")
            else:
                st.info("Filters reset")
            logger.info("All filters reset from expander")
            st.rerun()

    # --- END Filter Section inside an expander ---

    # Retrieve values from session state / widget values for filtering
    default_range = [df['waktu'].min().date(), df['waktu'].max().date()]
    date_range = st.session_state.get('date_range', default_range)
    start_date, end_date = date_range

    time_granularity = st.session_state.get('time_granularity', 'Bulanan')
    selected_products = st.session_state.get('selected_products', [])
    
    if 'kategori_produk' in df.columns:
        categories = sorted(df['kategori_produk'].unique())
        include_categories = st.session_state.get('include_categories', False)
        selected_categories = st.session_state.get('selected_categories', categories[0] if categories else None)
    else:
        include_categories = False
        selected_categories = None

    if 'metode_pembayaran' in df.columns:
        payment_methods = sorted(df['metode_pembayaran'].unique())
        enable_payment_filter = st.session_state.get('enable_payment_filter', False)
        selected_payment_method = st.session_state.get('selected_payment_method', payment_methods[0] if payment_methods else None)
    else:
        enable_payment_filter = False
        selected_payment_method = None

    if 'tipe_pesanan' in df.columns:
        order_types = sorted(df['tipe_pesanan'].unique())
        enable_order_type_filter = st.session_state.get('enable_order_type_filter', False)
        selected_order_type = st.session_state.get('selected_order_type', order_types[0] if order_types else None)
    else:
        enable_order_type_filter = False
        selected_order_type = None


    # Filter data with improved logic (this part remains outside the expander)
    with st.spinner("Memproses data..." if lang == "ID" else "Processing data..."):
        df_filtered = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(df_filtered['waktu']):
            df_filtered['waktu'] = pd.to_datetime(df_filtered['waktu'], errors='coerce')
            df_filtered.dropna(subset=['waktu'], inplace=True)

        filters_applied = False

        if start_date and end_date: # Check if date_range is valid
            df_filtered = df_filtered[(df_filtered['waktu'].dt.date >= start_date) & 
                                      (df_filtered['waktu'].dt.date <= end_date)]
            filters_applied = True

        if selected_products:
            df_filtered = df_filtered[df_filtered['nama_produk'].isin(selected_products)]
            filters_applied = True
        if include_categories and selected_categories and 'kategori_produk' in df.columns:
            df_filtered = df_filtered[df_filtered['kategori_produk'] == selected_categories]
            filters_applied = True
        
        if enable_payment_filter and selected_payment_method and 'metode_pembayaran' in df.columns:
            df_filtered = df_filtered[df_filtered['metode_pembayaran'] == selected_payment_method]
            filters_applied = True
        
        if enable_order_type_filter and selected_order_type and 'tipe_pesanan' in df.columns:
            df_filtered = df_filtered[df_filtered['tipe_pesanan'] == selected_order_type]
            filters_applied = True

        if filters_applied and df_filtered.empty:
            if lang == "ID":
                st.markdown("""
                <div class="st-alert st-alert-error">
                    <b>⚠️ Tidak ada data!</b><br>
                    Filter yang dipilih tidak menghasilkan data. Silakan sesuaikan filter atau klik 'Reset Semua Filter'.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="st-alert st-alert-error">
                    <b>⚠️ No data available!</b><br>
                    The selected filters resulted in no data. Please adjust the filters or click 'Reset All Filters'.
                </div>
                """, unsafe_allow_html=True)
            st.stop()

    # 1. Sales Summary
    st.subheader("📊 Ringkasan Penjualan" if lang == "ID" else "Sales Summary")

    total_sales = df_filtered['total_pembayaran'].sum()
    total_transactions = len(df_filtered['id_transaksi'].unique())
    total_products_sold = df_filtered['jumlah'].sum()
    
    if time_granularity == "Harian":
        daily_transactions_sales = df_filtered.drop_duplicates(subset='id_transaksi').groupby(df_filtered['waktu'].dt.date)['total_pembayaran'].sum()
        avg_sales_per_period = daily_transactions_sales.mean() if not daily_transactions_sales.empty else 0
    elif time_granularity == "Mingguan":
        weekly_transactions_sales = df_filtered.drop_duplicates(subset='id_transaksi').groupby(df_filtered['waktu'].dt.isocalendar().week.astype(str) + '-' + df_filtered['waktu'].dt.year.astype(str))['total_pembayaran'].sum()
        avg_sales_per_period = weekly_transactions_sales.mean() if not weekly_transactions_sales.empty else 0
    elif time_granularity == "Bulanan":
        monthly_transactions_sales = df_filtered.drop_duplicates(subset='id_transaksi').groupby(df_filtered['waktu'].dt.to_period('M').astype(str))['total_pembayaran'].sum()
        avg_sales_per_period = monthly_transactions_sales.mean() if not monthly_transactions_sales.empty else 0
    else:
        yearly_transactions_sales = df_filtered.drop_duplicates(subset='id_transaksi').groupby(df_filtered['waktu'].dt.year)['total_pembayaran'].sum()
        avg_sales_per_period = yearly_transactions_sales.mean() if not yearly_transactions_sales.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{"Total Penjualan" if lang == "ID" else "Total Sales"}</div>
            <div class="metric-value">Rp {total_sales:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{"Jumlah transaksi" if lang == "ID" else "Total Transactions"}</div>
            <div class="metric-value">{total_transactions:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{"Produk Terjual (Unit)" if lang == "ID" else "Products Sold (Units)"}</div>
            <div class="metric-value">{total_products_sold:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_sales_label = ""
        if time_granularity == "Harian": avg_sales_label = "Rata-rata Penjualan Harian" if lang == "ID" else "Avg. Daily Sales"
        elif time_granularity == "Mingguan": avg_sales_label = "Rata-rata Penjualan Mingguan" if lang == "ID" else "Avg. Weekly Sales"
        elif time_granularity == "Bulanan": avg_sales_label = "Rata-rata Penjualan Bulanan" if lang == "ID" else "Avg. Monthly Sales"
        elif time_granularity == "Tahunan": avg_sales_label = "Rata-rata Penjualan Tahunan" if lang == "ID" else "Avg. Yearly Sales"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{avg_sales_label}</div>
            <div class="metric-value">Rp {avg_sales_per_period:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Product Performance
    st.subheader("🛒 Performa Produk" if lang == "ID" else "Product Performance")
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            top_products = df_filtered.groupby('nama_produk')['jumlah'].sum().sort_values(ascending=False).head(5).reset_index()
            fig = px.bar(top_products, x='jumlah', y='nama_produk', orientation='h',
                         title="5 Produk Terlaris (Unit)" if lang == "ID" else "Top 5 Products (Units Sold)",
                         labels={'nama_produk': 'Produk' if lang == "ID" else 'Product', 'jumlah': 'Unit Terjual' if lang == "ID" else 'Units Sold'},
                         template=plotly_template, color_discrete_sequence=['#4169E1'])
            fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20), yaxis={'categoryorder':'total ascending'})
            fig.update_traces(hovertemplate='%{y}<br>Units: %{x:,.0f}')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top_revenue_products = df_filtered.groupby('nama_produk')['harga_setelah_pajak'].sum().sort_values(ascending=False).head(5).reset_index()
            fig_rev = px.bar(top_revenue_products, x='harga_setelah_pajak', y='nama_produk', orientation='h',
                         title="5 Produk dengan Pendapatan Tertinggi" if lang == "ID" else "Top 5 Products by Revenue",
                         labels={'nama_produk': 'Produk' if lang == "ID" else 'Product', 'harga_setelah_pajak': 'Pendapatan (Rp)' if lang == "ID" else 'Revenue (Rp)'},
                         template=plotly_template, color_discrete_sequence=['#28a745'])
            fig_rev.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20), yaxis={'categoryorder':'total ascending'})
            fig_rev.update_traces(hovertemplate='%{y}<br>Revenue: Rp %{x:,.0f}')
            st.plotly_chart(fig_rev, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Sales Trends
    st.subheader("📈 Tren Penjualan" if lang == "ID" else "Sales Trends")
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        df_unique_transactions = df_filtered.drop_duplicates(subset='id_transaksi')

        if time_granularity == "Harian":
            sales_over_time = df_unique_transactions.groupby(df_unique_transactions['waktu'].dt.date)['total_pembayaran'].sum().reset_index()
            sales_over_time.columns = ['waktu', 'total_pembayaran']
            x_axis = 'waktu'
            x_title = 'Tanggal' if lang == "ID" else 'Date'
        elif time_granularity == "Mingguan":
            sales_over_time = df_unique_transactions.groupby(df_unique_transactions['waktu'].dt.isocalendar().week.astype(str) + '-' + df_unique_transactions['waktu'].dt.year.astype(str))['total_pembayaran'].sum().reset_index()
            sales_over_time.columns = ['waktu', 'total_pembayaran']
            x_axis = 'waktu'
            x_title = 'Minggu-Tahun' if lang == "ID" else 'Week-Year'
        elif time_granularity == "Bulanan":
            sales_over_time = df_unique_transactions.groupby(df_unique_transactions['waktu'].dt.to_period('M').astype(str))['total_pembayaran'].sum().reset_index()
            sales_over_time.columns = ['waktu', 'total_pembayaran']
            x_axis = 'waktu'
            x_title = 'Bulan-Tahun' if lang == "ID" else 'Month-Year'
        else:
            sales_over_time = df_unique_transactions.groupby(df_unique_transactions['waktu'].dt.year)['total_pembayaran'].sum().reset_index()
            sales_over_time.columns = ['waktu', 'total_pembayaran']
            x_axis = 'waktu'
            x_title = 'Tahun' if lang == "ID" else 'Year'
        
        fig = px.line(sales_over_time, x=x_axis, y='total_pembayaran',
                      title="Tren Penjualan dari Waktu ke Waktu" if lang == "ID" else "Sales Trend Over Time",
                      labels={x_axis: x_title, 'total_pembayaran': 'Penjualan (Rp)' if lang == "ID" else 'Sales (Rp)'},
                      template=plotly_template, color_discrete_sequence=['#4169E1'])
        fig.update_traces(marker=dict(size=6), line=dict(width=1.5), hovertemplate=f'{x_title}: %{{x}}<br>Sales: Rp %{{y:,.0f}}')
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Category and Transaction Analysis
    st.subheader("📋 Analisis Kategori & Transaksi" if lang == "ID" else "Category & Transaction Analysis")
    
    col_cat, col_trans_dist = st.columns(2)

    with col_cat:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if 'kategori_produk' in df_filtered.columns:
                category_sales = df_filtered.groupby('kategori_produk')['total_pembayaran'].sum().sort_values(ascending=True).reset_index()
                
                fig_cat = px.bar(category_sales, 
                                 y='kategori_produk', 
                                 x='total_pembayaran', 
                                 orientation='h',
                                 title="Pendapatan per Kategori" if lang == "ID" else "Revenue by Category",
                                 labels={'kategori_produk': 'Kategori' if lang == "ID" else 'Category', 'total_pembayaran': 'Pendapatan (Rp)' if lang == "ID" else 'Revenue (Rp)'},
                                 template=plotly_template,
                                 color_discrete_sequence=['#ff7f0e'])
                fig_cat.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
                fig_cat.update_traces(hovertemplate='Category: %{y}<br>Revenue: Rp %{x:,.0f}')
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Kolom 'kategori_produk' tidak ditemukan." if lang == "ID" else "'kategori_produk' column not found.")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_trans_dist:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            tab_payment, tab_order = st.tabs(["Metode Pembayaran" if lang == "ID" else "Payment Method", "Tipe Pesanan" if lang == "ID" else "Order Type"])

            with tab_payment:
                if 'metode_pembayaran' in df_filtered.columns:
                    payment_dist = df_filtered['metode_pembayaran'].value_counts().reset_index()
                    payment_dist.columns = ['metode', 'jumlah']
                    
                    fig_payment = go.Figure(data=[go.Pie(labels=payment_dist['metode'], 
                                                         values=payment_dist['jumlah'], 
                                                         hole=.4,
                                                         marker_colors=px.colors.sequential.Blues_r)])
                    fig_payment.update_layout(title_text="Distribusi Metode Pembayaran" if lang == "ID" else "Payment Method Distribution",
                                              template=plotly_template,
                                              showlegend=True,
                                              margin=dict(l=20, r=20, t=40, b=20))
                    fig_payment.update_traces(hovertemplate='%{label}: %{value} (%{percent})')
                    st.plotly_chart(fig_payment, use_container_width=True)
                else:
                    st.info("Kolom 'metode_pembayaran' tidak ditemukan." if lang == "ID" else "'metode_pembayaran' column not found.")

            with tab_order:
                if 'tipe_pesanan' in df_filtered.columns:
                    order_type_dist = df_filtered['tipe_pesanan'].value_counts().reset_index()
                    order_type_dist.columns = ['tipe', 'jumlah']

                    fig_order = go.Figure(data=[go.Pie(labels=order_type_dist['tipe'], 
                                                       values=order_type_dist['jumlah'], 
                                                       hole=.4,
                                                       marker_colors=px.colors.sequential.Greens_r)])
                    fig_order.update_layout(title_text="Distribusi Tipe Pesanan" if lang == "ID" else "Order Type Distribution",
                                            template=plotly_template,
                                            showlegend=True,
                                            margin=dict(l=20, r=20, t=40, b=20))
                    fig_order.update_traces(hovertemplate='%{label}: %{value} (%{percent})')
                    st.plotly_chart(fig_order, use_container_width=True)
                else:
                    st.info("Kolom 'tipe_pesanan' tidak ditemukan." if lang == "ID" else "'tipe_pesanan' column not found.")
            st.markdown('</div>', unsafe_allow_html=True)

    # Advanced Mode: Enhanced Features
    if mode == 'Advanced':
        st.subheader("🔍 Validasi Data" if lang == "ID" else "Data Validation")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            col_val1, col_val2, col_val3 = st.columns(3)
            with col_val1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{"Total Baris" if lang == "ID" else "Total Rows"}</div>
                    <div class="metric-value">{len(df_filtered):,}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_val2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{"Nilai Kosong" if lang == "ID" else "Missing Values"}</div>
                    <div class="metric-value">{df_filtered.isnull().sum().sum():,}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_val3:
                duplicates = df_filtered.duplicated(subset=['id_transaksi', 'waktu', 'nama_produk', 'tipe_pesanan']).sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{"Duplikat" if lang == "ID" else "Duplicates"}</div>
                    <div class="metric-value">{duplicates:,}</div>
                </div>
                """, unsafe_allow_html=True)
            if duplicates > 0:
                if st.button("Hapus Duplikat" if lang == "ID" else "Remove Duplicates", help="Hapus baris duplikat berdasarkan id_transaksi, waktu, nama_produk, dan tipe_pesanan" if lang == "ID" else "Remove duplicate rows based on id_transaksi, waktu, nama_produk, and tipe_pesanan"):
                    df_filtered = df_filtered.drop_duplicates(subset=['id_transaksi', 'waktu', 'nama_produk', 'tipe_pesanan'])
                    st.session_state.df = df_filtered
                    if lang == "ID":
                        st.success("Duplikat telah dihapus")
                    else:
                        st.success("Duplicates have been removed")
                    logger.info("Duplicates removed from filtered data")
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Correlation Heatmap
        st.subheader("📊 Korelasi Data" if lang == "ID" else "Data Correlation")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if 'id_transaksi' not in col.lower()]

            if len(numeric_cols) > 1:
                plt.figure(figsize=(8, 6))
                correlation_matrix = df_filtered[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt=".2f")
                plt.title("Korelasi Antara Variabel Numerik" if lang == "ID" else "Correlation Between Numeric Variables", color='white')
                plt.xticks(color='white', rotation=45, ha='right')
                plt.yticks(color='white', rotation=0)
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                st.image(buf, use_container_width=True)
                buf.close()
                plt.close(fig=plt.gcf())
            else:
                if lang == "ID":
                    st.warning("Tidak cukup data numerik untuk heatmap korelasi.")
                else:
                    st.warning("Not enough numeric data for correlation heatmap.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Outlier Detection with Professional Visualization
        st.subheader("🚨 Deteksi Outlier" if lang == "ID" else "Outlier Detection")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if 'id_transaksi' not in col.lower() and 'tahun' not in col.lower() and 'bulan' not in col.lower()]

            if len(numeric_cols) > 0:
                fig = go.Figure()
                outlier_counts = []
                for col in numeric_cols:
                    if not df_filtered[col].empty and pd.api.types.is_numeric_dtype(df_filtered[col]):
                        Q1 = df_filtered[col].quantile(0.25)
                        Q3 = df_filtered[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = df_filtered[(df_filtered[col] < lower_bound) | (df_filtered[col] > upper_bound)][col]
                        outlier_counts.append(len(outliers))
                        
                        fig.add_trace(go.Box(y=df_filtered[col], name=col, boxpoints='outliers', jitter=0.3, pointpos=-1.8, marker_color='#4169E1'))
                    else:
                        outlier_counts.append(0)
                
                fig.update_layout(
                    title="Distribusi Outlier pada Kolom Numerik" if lang == "ID" else "Outlier Distribution Across Numeric Columns",
                    yaxis_title="Nilai" if lang == "ID" else "Value",
                    template=plotly_template,
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                outlier_summary = pd.DataFrame({
                    "Kolom" if lang == "ID" else "Column": numeric_cols,
                    "Jumlah Outlier" if lang == "ID" else "Outlier Count": outlier_counts
                })
                st.dataframe(outlier_summary.style.format({"Jumlah Outlier" if lang == "ID" else "Outlier Count": "{:,}"}))
                
                total_outliers = sum(outlier_counts)
                if total_outliers == 0:
                    if lang == "ID":
                        st.success("Tidak ada outlier yang terdeteksi.")
                    else:
                        st.success("No outliers detected.")
                else:
                    if lang == "ID":
                        st.warning(f"Total {total_outliers:,} outlier(s) terdeteksi di seluruh kolom.")
                    else:
                        st.warning(f"Total {total_outliers:,} outlier(s) detected across all columns.")
            else:
                if lang == "ID":
                    st.warning("Tidak ada data numerik yang cocok untuk deteksi outlier.")
                else:
                    st.warning("No suitable numeric data for outlier detection.")
            st.markdown('</div>', unsafe_allow_html=True)

    # Download Data
    st.download_button(
        label="💾 Unduh Data" if lang == "ID" else "Download Data",
        data=df_filtered.to_csv(index=False, sep=';', decimal=',').encode('utf-8'),
        file_name="data_penjualan_filtered.csv",
        mime='text/csv'
    )