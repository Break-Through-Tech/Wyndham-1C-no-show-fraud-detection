
import pandas as pd
from pathlib import Path

print("\n" + "=" * 80)
print("FEATURE VALIDATION AND GENERATION")
print("=" * 80)

# CONFIGURATION
REQUIRED_FEATURES = [
    "cancellation_lead_time_days",
    "booking_day_of_month",
    "booking_lead_time_days",
    "member_tenure_days_fraud",
    "is_member",
    "stay_length"
]

# Input files to process
INPUT_FILES = [
    "merged_may_2026.csv",
    "merged_june_2026.csv",
    "merged_july_2026.csv"
]

# Date columns used in feature generation
DATE_COLUMNS = [
    "booking_timestamp",
    "booking_date",
    "cancellation_date",
    "check_in_date",
    "check_out_date",
    "member_enrollment_date"
]



# HELPER FUNCTIONS
def ensure_datetime(data, column):
    """Convert a specific column to datetime if it exists."""

    if column in data.columns:
        data[column] = pd.to_datetime(
            data[column],
            errors="coerce",
            utc=True
        ).dt.tz_localize(None)


def record_status(report, feature, status):
    """Record the generation status of a feature."""

    report.append({
        "feature": feature,
        "status": status
    })

# PROCESS EACH DATASET
all_reports = []

for input_file in INPUT_FILES:

    print("\n" + "=" * 80)
    print(f"PROCESSING: {input_file}")
    print("=" * 80)

    df = pd.read_csv(input_file)

    # Separate report for each dataset
    feature_generation_report = []

    # Convert available date columns
    for col in DATE_COLUMNS:
        ensure_datetime(df, col)
    
    # 1. CANCELLATION LEAD TIME
    # Sources: cancellation_date, check_in_date
    if "cancellation_lead_time_days" not in df.columns:

        if "cancellation_date" in df.columns and "check_in_date" in df.columns:

            df["cancellation_lead_time_days"] = (
                df["check_in_date"] - df["cancellation_date"]
            ).dt.days

            status = "Generated from check_in_date and cancellation_date"

        else:
            status = (
                "Unavailable: cancellation_date or check_in_date "
                "column does not exist"
            )

    else:
        status = "Already exists"

    record_status(
        feature_generation_report,
        "cancellation_lead_time_days",
        status
    )


    
    # 2. BOOKING DAY OF MONTH
    # Source: booking_timestamp
    if "booking_day_of_month" not in df.columns:

        if "booking_timestamp" in df.columns:

            df["booking_day_of_month"] = (
                df["booking_timestamp"].dt.day
            )

            status = "Generated from booking_timestamp"

        else:
            status = (
                "Unavailable: booking_timestamp column does not exist"
            )

    else:
        status = "Already exists"

    record_status(
        feature_generation_report,
        "booking_day_of_month",
        status
    )


    
    # 3. BOOKING LEAD TIME
    # Sources: booking_timestamp, check_in_date
    if "booking_lead_time_days" not in df.columns:

        if "booking_timestamp" in df.columns and "check_in_date" in df.columns:

            df["booking_lead_time_days"] = (
                df["check_in_date"]
                - df["booking_timestamp"].dt.normalize()
            ).dt.days

            status = "Generated from booking_timestamp and check_in_date"

        else:
            status = (
                "Unavailable: booking_timestamp or check_in_date "
                "column does not exist"
            )

    else:
        status = "Already exists"

    record_status(
        feature_generation_report,
        "booking_lead_time_days",
        status
    )


    
    # 4. MEMBER TENURE DAYS
    # Sources: booking_timestamp, member_enrollment_date
    if "member_tenure_days_fraud" not in df.columns:

        if "booking_timestamp" in df.columns and "member_enrollment_date" in df.columns:

            df["member_tenure_days_fraud"] = (
                df["booking_timestamp"].dt.normalize()
                - df["member_enrollment_date"]
            ).dt.days

            status = (
                "Generated from booking_timestamp "
                "and member_enrollment_date"
            )

        else:
            status = (
                "Unavailable: booking_timestamp or "
                "member_enrollment_date column does not exist"
            )

    else:
        status = "Already exists"

    record_status(
        feature_generation_report,
        "member_tenure_days_fraud",
        status
    )


    
    # 5. IS MEMBER
    # Source: member_enrollment_date
    if "is_member" not in df.columns:

        if "member_enrollment_date" in df.columns:

            df["is_member"] = (
                df["member_enrollment_date"].notna().astype(int)
            )

            status = "Generated from member_enrollment_date"

        else:
            df["is_member"] = 0

            status = (
                "Generated as 0: member_enrollment_date "
                "column does not exist"
            )

    else:
        status = "Already exists"

    record_status(
        feature_generation_report,
        "is_member",
        status
    )


    
    # 6. STAY LENGTH
    # Sources: check_in_date, check_out_date
    if "check_in_date" in df.columns and "check_out_date" in df.columns:

        df["stay_length"] = (
            df["check_out_date"] - df["check_in_date"]
        ).dt.days

        status = "Generated from check_out_date and check_in_date"

    else:
        status = (
            "Unavailable: check_in_date or check_out_date "
            "column does not exist"
        )

    record_status(
        feature_generation_report,
        "stay_length",
        status
    )


    
    # VALIDATE DERIVED DATE FEATURES
    derived_date_features = [
        "cancellation_lead_time_days",
        "booking_lead_time_days",
        "member_tenure_days_fraud",
        "stay_length"
    ]

    for col in derived_date_features:

        if col in df.columns:

            negative_count = (
                pd.to_numeric(df[col], errors="coerce") < 0
            ).sum()

            if negative_count > 0:
                print(
                    f"WARNING: {col} contains "
                    f"{negative_count:,} negative values."
                )


    
    # FEATURE GENERATION REPORT
    report_df = pd.DataFrame(feature_generation_report)

    print("\nFeature generation report:")
    print(report_df.to_string(index=False))

    # Check required features
    missing_features = [
        col for col in REQUIRED_FEATURES
        if col not in df.columns
    ]

    if missing_features:

        print("\nFeatures that could not be generated:")

        for col in missing_features:
            print(f"  - {col}")

    else:
        print("\nAll required features are present.")


    
    # SAVE INDIVIDUAL FEATURE REPORT
    input_path = Path(input_file)

    report_filename = (
        f"{input_path.stem}_feature_generation_report.csv"
    )

    report_df.to_csv(report_filename, index=False)

    # Add dataset identifier for combined reporting
    report_df.insert(0, "dataset", input_path.name)

    all_reports.append(report_df)

    
    
    # 7. CLEAN IS_FRAUD_EVENT
    if "is_fraud_event" in df.columns:

        # Replace empty strings and whitespace with 0
        df["is_fraud_event"] = df["is_fraud_event"].replace(
            r"^\s*$",
            0,
            regex=True
        )

        # Replace missing values with 0
        df["is_fraud_event"] = df["is_fraud_event"].fillna(0)

        # Convert to integer
        df["is_fraud_event"] = df["is_fraud_event"].astype(int)

        status = (
            "Cleaned: empty strings and NaN replaced with 0; "
            "converted to integer"
        )

    else:
        status = "Unavailable: is_fraud_event column does not exist"

    record_status(
        feature_generation_report,
        "is_fraud_event",
        status
    )


    
    # SAVE ENRICHED DATASET
    output_filename = f"{input_path.stem}_enriched.csv"

    df.to_csv(output_filename, index=False)

    print(f"\nSaved enriched dataset: {output_filename}")
    print(f"Final dataset shape: {df.shape}")

    # Stay length summary
    if "stay_length" in df.columns:
        print("\nStay length summary:")
        print(df["stay_length"].describe())



# SAVE COMBINED FEATURE GENERATION REPORT
combined_report = pd.concat(
    all_reports,
    ignore_index=True
)

combined_report.to_csv(
    "all_feature_generation_reports.csv",
    index=False
)

print("\n" + "=" * 80)
print("ALL DATASETS PROCESSED")
print("=" * 80)

print("\nCombined feature generation report saved.")



# MERGE MAY, JUNE AND JULY ENRICHED DATASETS
print("\n" + "=" * 80)
print("MERGING ALL THREE ENRICHED DATASETS")
print("=" * 80)

# Input files
files = [
    "merged_may_2026_enriched.csv",
    "merged_june_2026_enriched.csv",
    "merged_july_2026_enriched.csv"
]

# Read all three CSV files
dfs = [pd.read_csv(f) for f in files]

# Concatenate vertically
combined = pd.concat(
    dfs,
    ignore_index=True
)

# Save combined dataset
combined_filename = "merged_may_june_july_2026_enriched.csv"

combined.to_csv(
    combined_filename,
    index=False
)


# VALIDATE COMBINED DATASET
print("\nCombined dataset successfully created!")

print(f"Output file: {combined_filename}")

print("\nIndividual dataset row counts:")

for file, df_month in zip(files, dfs):
    print(f"{file}: {len(df_month):,} rows")

print("\nCombined dataset information:")
print(f"Total rows: {len(combined):,}")
print(f"Total columns: {len(combined.columns):,}")

print("\nCombined dataset shape:")
print(combined.shape)

# Check required features
print("\nChecking required features:")

for col in REQUIRED_FEATURES:
    if col in combined.columns:
        print(f"  [OK] {col}")
    else:
        print(f"  [MISSING] {col}")

# Check is_fraud_event
if "is_fraud_event" in combined.columns:
    print("\nis_fraud_event value counts:")
    print(combined["is_fraud_event"].value_counts(dropna=False))

# Check stay_length
if "stay_length" in combined.columns:
    print("\nStay length summary:")
    print(combined["stay_length"].describe())

# Check for missing values
print("\nColumns with missing values:")

missing_values = combined.isnull().sum()
missing_values = missing_values[missing_values > 0]

if len(missing_values) > 0:
    print(missing_values.sort_values(ascending=False))
else:
    print("No missing values found.")

print("\n" + "=" * 80)
print("ALL THREE MONTHS SUCCESSFULLY MERGED")
print("=" * 80)

print(f"\nFinal combined file: {combined_filename}")
print(f"Total records: {len(combined):,}")

