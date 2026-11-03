
import pandas as pd

months = ["may", "june", "july"]

# Loop through each month and merge the corresponding files
for month in months:

    print(f"\nProcessing {month.capitalize()} 2026...")

    # File paths
    stays_path = f"stays_{month}_2026.csv"
    fraud_path = f"fraud_event_ground_truth_{month}2026.csv"
    reservations_path = f"reservations_{month}_2026.csv"

    stays = pd.read_csv(stays_path)

    # Keep only the needed columns from the fraud file
    fraud = pd.read_csv(
        fraud_path,
        usecols=["confirmation_number", "is_fraud_event"]
    )

    reservations = pd.read_csv(reservations_path)

    # Identify overlapping columns (excluding the join key)
    overlap_cols = [
        col for col in reservations.columns
        if col in stays.columns and col != "confirmation_number"
    ]

    # Keep only non-overlapping columns plus the confirmation number
    reservations_clean = reservations.drop(columns=overlap_cols)

    # 5a – Add the fraud flag
    merged = pd.merge(
        stays,
        fraud,
        on="confirmation_number",
        how="left",
        suffixes=("", "")
    )

    # 5b – Add reservation data (extra columns only)
    merged = pd.merge(
        merged,
        reservations_clean,
        on="confirmation_number",
        how="left",
        suffixes=("", "")
    )

    merged = merged.drop_duplicates()

    # 6. Save file
    output_path = f"merged_{month}_2026.csv"

    merged.to_csv(output_path, index=False)

    print(f"Merged file created: {output_path}")

print("\nAll three monthly files have been successfully processed!")
