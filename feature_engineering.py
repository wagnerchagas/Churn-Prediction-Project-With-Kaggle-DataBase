# ============================
# === FEATURE ENGINEERING ===
# ============================

def add_tenure_group(df):
    """
    Create tenure group category to capture customer lifecycle stages.
    """
    def group(x):
        if x <= 6:
            return '0-6'
        elif x <= 12:
            return '7-12'
        elif x <= 24:
            return '13-24'
        elif x <= 48:
            return '25-48'
        else:
            return '49+'
    df['tenure_group'] = df['tenure'].apply(group)
    return df


def add_avg_charge_per_month(df):
    """
    Calculate average charge per month of service.
    Useful to identify short-term customers with high cost.
    """
    df['avg_charge_per_month'] = df['TotalCharges'] / (df['tenure'] + 1e-5)
    return df


def add_services_count(df):
    """
    Count the number of services contracted by the customer.
    Customers with more services tend to be more engaged and loyal.
    """
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['services_count'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)
    return df


def add_is_fiber_optic_only(df):
    """
    Flag customers who only have fiber optic internet with no other service.
    May indicate dissatisfaction due to price vs. perceived value.
    """
    df['is_fiber_optic_only'] = (
        (df['InternetService'] == 'Fiber optic') &
        (df['OnlineSecurity'] == 'No') &
        (df['OnlineBackup'] == 'No') &
        (df['DeviceProtection'] == 'No') &
        (df['TechSupport'] == 'No')
    ).astype(int)
    return df


def add_has_streaming_and_no_security(df):
    """
    Flag customers who pay for streaming but have no online security.
    Could represent unaware or risky profiles.
    """
    df['has_streaming_and_no_security'] = (
        ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')) &
        (df['OnlineSecurity'] == 'No')
    ).astype(int)
    return df


def add_is_paperless_electronic(df):
    """
    Combine paperless billing and electronic check payment.
    This combination may correlate with churn-prone behavior.
    """
    df['is_paperless_electronic'] = (
        (df['PaperlessBilling'] == 'Yes') &
        (df['PaymentMethod'] == 'Electronic check')
    ).astype(int)
    return df


def add_senior_and_alone(df):
    """
    Identify senior citizens who live alone (no partner or dependents).
    Often associated with lower engagement and higher churn risk.
    """
    df['senior_and_alone'] = (
        (df['SeniorCitizen'] == 1) &
        (df['Partner'] == 'No') &
        (df['Dependents'] == 'No')
    ).astype(int)
    return df


def add_price_sensitive(df):
    """
    Identify price-sensitive customers: high monthly charge relative to tenure.
    These users may be more likely to churn due to cost dissatisfaction.
    """
    df['price_sensitive'] = (df['avg_charge_per_month'] > 90).astype(int)
    return df


# === APPLY ALL CUSTOM FEATURES ===

def apply_custom_features(df):
    """
    Apply all domain-based engineered features to the dataset.
    """
    df = add_tenure_group(df)
    df = add_avg_charge_per_month(df)
    df = add_services_count(df)
    df = add_is_fiber_optic_only(df)
    df = add_has_streaming_and_no_security(df)
    df = add_is_paperless_electronic(df)
    df = add_senior_and_alone(df)
    df = add_price_sensitive(df)
    return df
