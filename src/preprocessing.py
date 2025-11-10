import pandas as pd
import os

def load_and_clean_iotid20(input_csv, output_csv):
    # Columns to drop (from Table 7 in the paper)
    drop_cols = [
        'Flow_ID', 'Src_IP', 'Src_Port', 'Dst_IP', 'Dst_Port', 'Timestamp',
        'Label', 'Sub_Cat', 'Cat'
    ]
    df = pd.read_csv(input_csv)
    # Filter to DoS and Normal only (using Cat column)
    df = df[df['Cat'].isin(['DoS', 'Normal'])].copy()
    # Encode labels before dropping: DoS=1, Normal=0
    df['Target'] = df['Cat'].map({'DoS': 1, 'Normal': 0})
    # Drop identifier columns if present
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    # Save cleaned CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned IoTID20 to {output_csv} with shape {df.shape}")
    return df

