create TABLE tenants {
    tenant_id VARCHAR(50) PRIMARY KEY,
    tenant_name VARCHAR(200) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    subscription_tier VARCHAR(20) DEFAULT 'free'
}

CREATE TABLE stores (
    store_id VARCHAR(50) PRIMARY KEY,
    city_id VARCHAR(50),
    store_name VARCHAR(200),
    opened_date DATE
);

CREATE TABLE products (
    product_id VARCHAR(50) PRIMARY KEY,
    management_group_id VARCHAR(50),
    first_category_id VARCHAR(50),
    second_category_id VARCHAR(50),
    third_category_id VARCHAR(50),
);

Create Table sales (
    sales_id VARCHAR(50) PRIMARY KEY,
    tenant_id VARCHAR(50),
    store_id VARCHAR(50),
    product_id VARCHAR(50),
    sales_date DATE,
    sales_quantity INT,
    sales_amount DECIMAL(10, 2)
);

CREATE TABLE stockout_events (
    stockout_id SERIAL PRIMARY KEY,
    store_id VARCHAR(50),
    product_id VARCHAR(50),
    stockout_date DATE,
    stockout_hour INTEGER,
    hours_stock_status LIST<int>,
    lost_sales_estimate DECIMAL(10, 2),
);

