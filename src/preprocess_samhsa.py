import pandas as pd

INPUT_CSV = "data/boston_samhsa_raw.csv"
OUTPUT_CSV = "data/boston_samhsa_clean.csv"

KEEP_COLS = [
    # basic info
    "name1", "name2", "street1", "street2", "city", "state",
    "zip", "county", "phone", "website", "latitude", "longitude",
    "type_facility",

    # core need type
    "mh", "sa",

    # care level
    "tele", "res", "php", "op", "iop", "ipt",

    # payment
    "cash", "md", "mc", "pi", "tricare", "swfs",

    # treatment features
    "cbt", "dbt", "otp", "pmat",

    # populations
    "vet", "adol", "pw",

    # language / accessibility
    "asl", "sp"
]

RENAME_MAP = {
    "name1": "facility_name",
    "name2": "program_name",
    "street1": "address1",
    "street2": "address2",
    "zip": "zip_code",

    "mh": "mental_health",
    "sa": "substance_use",

    "tele": "telehealth",
    "res": "residential",
    "php": "partial_hospitalization",
    "op": "outpatient",
    "iop": "intensive_outpatient",
    "ipt": "inpatient",

    "cash": "cash_payment",
    "md": "medicaid",
    "mc": "medicare",
    "pi": "private_insurance",
    "tricare": "tricare",
    "swfs": "sliding_fee_scale",

    "cbt": "cbt",
    "dbt": "dbt",
    "otp": "opioid_treatment_program",
    "pmat": "medication_assisted_treatment",

    "vet": "veterans",
    "adol": "adolescents",
    "pw": "pregnant_postpartum",

    "asl": "asl_support",
    "sp": "spanish_support"
}

FLAG_COLS = [
    "mental_health", "substance_use",
    "telehealth", "residential", "partial_hospitalization",
    "outpatient", "intensive_outpatient", "inpatient",
    "cash_payment", "medicaid", "medicare", "private_insurance",
    "tricare", "sliding_fee_scale",
    "cbt", "dbt", "opioid_treatment_program", "medication_assisted_treatment",
    "veterans", "adolescents", "pregnant_postpartum",
    "asl_support", "spanish_support"
]


def clean_text(series):
    return series.fillna("").astype(str).str.strip()


def convert_flag(series):
    """
    Convert columns like 1 / blank / NaN into boolean.
    """
    s = series.fillna("0").astype(str).str.strip()
    s = s.eq("1")
    return s


def build_full_address(row):
    parts = []
    if row["address1"]:
        parts.append(row["address1"])
    if row["address2"]:
        parts.append(row["address2"])
    city_state_zip = " ".join(
        x for x in [row["city"] + "," if row["city"] else "", row["state"], row["zip_code"]] if x
    ).strip()
    if city_state_zip:
        parts.append(city_state_zip)
    return ", ".join(parts)


def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)

    print("Raw shape:", df.shape)

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[KEEP_COLS].copy()
    df = df.rename(columns=RENAME_MAP)

    text_cols = [
        "facility_name", "program_name", "address1", "address2",
        "city", "state", "zip_code", "county", "phone", "website",
        "latitude", "longitude", "type_facility"
    ]
    for col in text_cols:
        df[col] = clean_text(df[col])

    for col in FLAG_COLS:
        df[col] = convert_flag(df[col])

    df["full_address"] = df.apply(build_full_address, axis=1)

    df["display_name"] = df.apply(
        lambda row: row["facility_name"]
        if not row["program_name"]
        else f'{row["facility_name"]} — {row["program_name"]}',
        axis=1
    )

    # merge duplicate facility rows
    group_keys = ["facility_name", "address1", "city", "phone"]

    non_flag_cols = [
        "facility_name", "program_name", "address1", "address2",
        "city", "state", "zip_code", "county", "phone", "website",
        "latitude", "longitude", "type_facility", "full_address", "display_name"
    ]

    agg_map = {col: "max" for col in FLAG_COLS}
    for col in non_flag_cols:
        agg_map[col] = "first"

    df_clean = df.groupby(group_keys, dropna=False, as_index=False).agg(agg_map)

    print("Cleaned shape:", df_clean.shape)
    print(df_clean.head(5)[["display_name", "full_address", "mental_health", "substance_use", "outpatient", "telehealth", "medicaid"]])

    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned data to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()