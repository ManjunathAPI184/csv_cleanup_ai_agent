import streamlit as st
import pandas as pd
import io
import re
from collections import Counter

# --- Utility Functions ---

EMAIL_REGEX = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
PHONE_CHARS_REMOVE = re.compile(r"[^\d]")

def clean_header(header):
    return str(header).strip().lower().replace(" ", "_").replace("-", "_")

def normalize_email(email):
    if not isinstance(email, str) or email.strip() == "" or email.lower() == "nan":
        return "N/A"
    email = email.strip().lower()
    email = re.sub(r"@(gamil|gnail|gmial)\.", "@gmail.", email)
    email = re.sub(r"@(yahooo|yaho)\.", "@yahoo.", email)
    email = re.sub(r"@(hotnail|hotmial)\.", "@hotmail.", email)
    return email

def normalize_phone_number(num):
    if not isinstance(num, str) or num.strip() == "" or num.lower() == "nan":
        return "N/A"
    digits = PHONE_CHARS_REMOVE.sub("", num)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return digits if digits else "N/A"

def standardize_columns(df):
    column_mapping = {
        "name": "name", "full_name": "name", "customer_name": "name", "first_name": "name",
        "email": "email", "email_address": "email", "e_mail": "email", "e-mail": "email",
        "age": "age", "years_old": "age", "age_range": "age", "years": "age",
        "phone": "phone", "phone_number": "phone", "contact": "phone", "tel": "phone", "telephone": "phone",
        "notes": "notes"
    }
    new_columns = {}
    for col in df.columns:
        clean_col = clean_header(col)
        new_columns[col] = column_mapping.get(clean_col, clean_col)
    return df.rename(columns=new_columns)

def fill_and_strip(df, fill_value="N/A"):
    df = df.fillna(fill_value)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": fill_value, "None": fill_value})
    return df

def apply_normalizations(df, normalize_email_flag=True, normalize_phone_flag=True):
    if normalize_email_flag and "email" in df.columns:
        df["email"] = df["email"].apply(normalize_email)
    if normalize_phone_flag and "phone" in df.columns:
        df["phone"] = df["phone"].apply(normalize_phone_number)
    return df

def deduplicate(df, strategy="email"):
    if strategy == "email" and "email" in df.columns:
        return df.drop_duplicates(subset=["email"], keep="first")
    elif strategy == "phone" and "phone" in df.columns:
        return df.drop_duplicates(subset=["phone"], keep="first")
    elif strategy == "name_email" and all(c in df.columns for c in ["name", "email"]):
        return df.drop_duplicates(subset=["name", "email"], keep="first")
    elif strategy == "name_phone" and all(c in df.columns for c in ["name", "phone"]):
        return df.drop_duplicates(subset=["name", "phone"], keep="first")
    else:
        return df.drop_duplicates(keep="first")

# --- Enhanced Natural Language Parser ---
def parse_instructions(text):
    # Smart defaults that work for most cases
    params = {
        "fill_empty_with": "N/A",
        "sort_by": None,
        "filter_empty": [],
        "normalize_phone": True,
        "normalize_email": True,
        "keep_only_complete_rows": False,
        "dedup_strategy": "email"  # Best default for most business data
    }
    
    if not text:
        return params
    
    t = text.lower()
    
    # Fill missing values
    fill_match = re.search(r"fill (?:missing|empty)(?: values)? with ([^\n,]+)", t)
    if fill_match:
        params["fill_empty_with"] = fill_match.group(1).strip()
    elif "fill missing with unknown" in t:
        params["fill_empty_with"] = "Unknown"
    elif "fill missing with blank" in t or "leave empty" in t:
        params["fill_empty_with"] = ""
    
    # Sorting
    if "sort by age" in t or "order by age" in t:
        params["sort_by"] = "age"
    elif "sort by name" in t or "order by name" in t or "alphabetical" in t:
        params["sort_by"] = "name"
    
    # Remove specific empty rows
    if "remove rows where email is empty" in t or "delete empty emails" in t:
        params["filter_empty"].append("email")
    if "remove rows where phone is empty" in t or "delete empty phones" in t:
        params["filter_empty"].append("phone")
    if "remove rows where name is empty" in t or "delete empty names" in t:
        params["filter_empty"].append("name")
    
    # Keep only complete
    if any(phrase in t for phrase in ["only complete rows", "complete data only", "no empty fields", "no missing data"]):
        params["keep_only_complete_rows"] = True
    
    # Deduplication strategy (auto-detect best approach)
    if "dedup by phone" in t or "dedupe by phone" in t or "remove duplicate phones" in t:
        params["dedup_strategy"] = "phone"
    elif any(phrase in t for phrase in ["dedup by name and email", "dedupe by name and email", "remove duplicates by name+email"]):
        params["dedup_strategy"] = "name_email"
    elif any(phrase in t for phrase in ["dedup by name and phone", "dedupe by name and phone", "remove duplicates by name+phone"]):
        params["dedup_strategy"] = "name_phone"
    elif "remove all duplicates" in t or "dedup everything" in t:
        params["dedup_strategy"] = "all_columns"
    # Default stays as "email" - most common business case
    
    return params

def compute_metrics(df, original_len):
    metrics = []
    metrics.append(("Original Rows", original_len))
    metrics.append(("Final Rows", len(df)))
    metrics.append(("Rows Cleaned", original_len - len(df)))
    metrics.append(("Total Columns", len(df.columns)))
    
    # Quality metrics
    if "email" in df.columns:
        valid_emails = sum(1 for e in df["email"] if "@" in str(e) and str(e) != "N/A")
        metrics.append(("Valid Emails", valid_emails))
    
    if "phone" in df.columns:
        valid_phones = sum(1 for p in df["phone"] if len(re.sub(r"[^\d]", "", str(p))) >= 10 and str(p) != "N/A")
        metrics.append(("Valid Phones", valid_phones))
    
    return pd.DataFrame(metrics, columns=["Metric", "Value"])

def main():
    st.set_page_config(page_title="CSV Cleanup AI Agent", layout="centered")
    
    # Header with AI branding
    st.title("🤖 CSV Cleanup AI Agent")
    st.markdown("**Just tell me what you want in plain English - I'll handle the rest!**")
    
    # Examples in a nice format
    with st.expander("💡 See what I can do"):
        st.markdown("""
        **Just type things like:**
        - "Remove duplicates and fill missing emails with Unknown"
        - "Sort by age and remove rows where email is empty" 
        - "Fill missing with N/A and dedup by phone"
        - "Only complete rows with no missing data"
        - "Clean everything and sort alphabetically"
        - "Remove duplicate phones and fill empty with blank"
        """)
    
    uploaded_files = st.file_uploader(
        "📁 Upload your CSV files", 
        type=["csv"], 
        accept_multiple_files=True,
        help="Upload one or more CSV files to clean and merge"
    )
    
    # Main instruction input - prominent and friendly
    nl_instruction = st.text_area(
        "✨ Tell me what you want to do:",
        placeholder="e.g., Remove duplicates, fill missing emails with Unknown, sort by name",
        height=100,
        help="Describe in simple English what cleaning you need. I'll figure out the technical details!"
    )
    
    if st.button("🚀 Clean My Data", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("📤 Please upload at least one CSV file first!")
            return
        
        # Parse instructions with AI-like feedback
        params = parse_instructions(nl_instruction)
        
        # Show what the AI understood
        with st.expander("🧠 Here's what I understood from your request"):
            understood_actions = []
            if params["dedup_strategy"] == "email":
                understood_actions.append("Remove duplicate rows (by email)")
            elif params["dedup_strategy"] == "phone":
                understood_actions.append("Remove duplicate rows (by phone)")
            elif params["dedup_strategy"] == "name_email":
                understood_actions.append("Remove duplicates (by name + email)")
            elif params["dedup_strategy"] == "name_phone":
                understood_actions.append("Remove duplicates (by name + phone)")
            else:
                understood_actions.append("Remove exact duplicate rows")
                
            if params["fill_empty_with"]:
                understood_actions.append(f"Fill empty fields with '{params['fill_empty_with']}'")
                
            if params["sort_by"]:
                understood_actions.append(f"Sort data by {params['sort_by']}")
                
            if params["filter_empty"]:
                understood_actions.append(f"Remove rows with empty {', '.join(params['filter_empty'])}")
                
            if params["keep_only_complete_rows"]:
                understood_actions.append("Keep only rows with complete data")
                
            understood_actions.extend([
                "Standardize column names (Full Name → Name, etc.)",
                "Clean email formats and fix typos", 
                "Format phone numbers consistently"
            ])
            
            for action in understood_actions:
                st.write(f"• {action}")
        
        # Process files
        with st.spinner("🔄 Processing your data... This might take a moment."):
            dataframes = []
            total_original = 0
            
            for f in uploaded_files:
                try:
                    df = pd.read_csv(f)
                    dataframes.append(df)
                    total_original += len(df)
                except Exception as e:
                    st.error(f"❌ Couldn't read {f.name}: {e}")
                    return
            
            processed_dfs = []
            
            for df in dataframes:
                # Apply all cleaning steps
                df = standardize_columns(df)
                df = fill_and_strip(df, fill_value=params["fill_empty_with"])
                df = apply_normalizations(df, params["normalize_email"], params["normalize_phone"])
                
                # Filter rows with empty specific columns
                for col in params.get("filter_empty", []):
                    if col in df.columns:
                        df = df[df[col].astype(str).str.strip().ne("") & (df[col] != "N/A")]
                
                # Keep only complete rows if requested
                if params["keep_only_complete_rows"]:
                    mask_complete = df.apply(lambda x: all(x != "N/A" and str(x).strip() != ""), axis=1)
                    df = df[mask_complete]
                
                # Deduplicate
                df = deduplicate(df, strategy=params["dedup_strategy"])
                processed_dfs.append(df)
            
            # Merge all processed dataframes
            merged_df = pd.concat(processed_dfs, ignore_index=True)
            
            # Final deduplication across merged data
            merged_df = deduplicate(merged_df, strategy=params["dedup_strategy"])
            
            # Sort if requested
            if params.get("sort_by") and params["sort_by"] in merged_df.columns:
                merged_df = merged_df.sort_values(by=params["sort_by"], na_position="last")
        
        # Results presentation
        st.success(f"✅ **Done!** Your data is cleaned: {len(merged_df)} rows × {len(merged_df.columns)} columns")
        
        # Metrics in a clean format
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Rows", total_original)
        with col2:
            st.metric("Final Rows", len(merged_df))
        with col3:
            st.metric("Cleaned/Removed", total_original - len(merged_df))
        
        # Data preview
        st.subheader("📊 Your Cleaned Data")
        st.dataframe(merged_df.head(20), use_container_width=True)
        
        # Download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            merged_df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
            
            # Simple summary
            summary_data = {
                "Metric": ["Original Rows", "Final Rows", "Rows Cleaned", "Total Columns"],
                "Value": [total_original, len(merged_df), total_original - len(merged_df), len(merged_df.columns)]
            }
            pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name="Summary")
        
        st.download_button(
            label="📥 Download Your Cleaned Data (Excel)",
            data=output.getvalue(),
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )
        
        st.balloons()  # Celebration effect!

if __name__ == "__main__":
    main()
