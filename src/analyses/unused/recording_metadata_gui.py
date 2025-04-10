import os
import tkinter as tk
from datetime import date
from tkinter import ttk, messagebox

import pandas as pd

# CSV file name
CSV_FILE = "data_records.csv"

# Custom order of channels
channel_order = [
    15, 16, 1, 30, 8, 23, 0, 31, 14, 17, 2, 29, 13, 18, 7, 24,
    3, 28, 12, 19, 4, 27, 9, 22, 11, 20, 5, 26, 10, 21, 6, 25
]


# Reset all fields to defaults
def reset_to_defaults():
    date_entry.delete(0, tk.END)
    date_entry.insert(0, current_date)
    round_no_entry.delete(0, tk.END)
    target_ap_entry.delete(0, tk.END)
    target_ml_entry.delete(0, tk.END)
    target_dv_entry.delete(0, tk.END)
    absolute_depth_entry.delete(0, tk.END)
    azimuth_entry.delete(0, tk.END)
    elevation_entry.delete(0, tk.END)
    region_var.set("ER")
    for ch in channel_order:
        channel_vars[ch].set("No spikes")


# Save to CSV functionality
def save_to_csv():
    # Get input values
    date_value = date_entry.get()
    round_no = round_no_entry.get()
    target_ap = target_ap_entry.get()
    target_ml = target_ml_entry.get()
    target_dv = target_dv_entry.get()
    absolute_depth = absolute_depth_entry.get()
    azimuth = azimuth_entry.get()
    elevation = elevation_entry.get()
    region = region_var.get()

    # Collect channel data
    channel_data = {ch: channel_vars[ch].get() for ch in channel_order}

    # Validate inputs
    if not date_value or not round_no or not region:
        messagebox.showerror("Error", "Date, Round No, and Region are required!")
        return

    try:
        round_no = int(round_no)
        if round_no < 1 or round_no > 6:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Round No must be an integer between 1 and 6!")
        return

    try:
        target_ap = float(target_ap)
        target_ml = float(target_ml)
        target_dv = float(target_dv)
        absolute_depth = float(absolute_depth)
        azimuth = float(azimuth)
        elevation = float(elevation)
    except ValueError:
        messagebox.showerror("Error", "All location and angle fields must be valid floats!")
        return

    # Load existing data or create a new one
    if os.path.exists(CSV_FILE):
        data = pd.read_csv(CSV_FILE)
        # Ensure the expected columns exist
        if "Date" not in data.columns or "Round No" not in data.columns:
            messagebox.showerror("Error", "The CSV file is missing required columns. Please check the file.")
            return
    else:
        # Create a new DataFrame with required columns
        data = pd.DataFrame(columns=[
                                        "Date", "Round No", "Target AP Location", "Target ML Location",
                                        "Target DV Location", "Absolute Depth (µm)", "Azimuth (°)",
                                        "Elevation (°)", "Region"
                                    ] + [f"Channel {ch}" for ch in channel_order])

    # Check for duplicate Date and Round No
    duplicate = data[(data["Date"] == date_value) & (data["Round No"] == round_no)]

    if not duplicate.empty:
        if not messagebox.askyesno("Duplicate Entry", "An entry with the same Date and Round No exists. Overwrite?"):
            return
        data = data[~((data["Date"] == date_value) & (data["Round No"] == round_no))]

    # Prepare data for saving
    new_data = {
        "Date": date_value,
        "Round No": round_no,
        "Target AP Location": target_ap,
        "Target ML Location": target_ml,
        "Target DV Location": target_dv,
        "Absolute Depth (µm)": absolute_depth,
        "Azimuth (°)": azimuth,
        "Elevation (°)": elevation,
        "Region": region
    }
    # Add channel information to the data
    for ch, category in channel_data.items():
        new_data[f"Channel {ch}"] = category

    # Append new data and save
    new_data_df = pd.DataFrame([new_data])
    data = pd.concat([data, new_data_df], ignore_index=True)
    data.to_csv(CSV_FILE, index=False)
    messagebox.showinfo("Success", "Data saved successfully!")


# Create the main window
root = tk.Tk()
root.title("Data Entry Form")

# Labels and entry fields
tk.Label(root, text="Date (YYYY-MM-DD):").grid(row=0, column=0, padx=10, pady=5)

# Pre-fill current date
current_date = date.today().strftime("%Y-%m-%d")
date_entry = tk.Entry(root)
date_entry.insert(0, current_date)  # Set default to today's date
date_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Round No (1-6):").grid(row=1, column=0, padx=10, pady=5)
round_no_entry = tk.Entry(root)
round_no_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Target AP Location (float):").grid(row=2, column=0, padx=10, pady=5)
target_ap_entry = tk.Entry(root)
target_ap_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Target ML Location (float):").grid(row=3, column=0, padx=10, pady=5)
target_ml_entry = tk.Entry(root)
target_ml_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Target DV Location (float):").grid(row=4, column=0, padx=10, pady=5)
target_dv_entry = tk.Entry(root)
target_dv_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Absolute Depth (µm):").grid(row=5, column=0, padx=10, pady=5)
absolute_depth_entry = tk.Entry(root)
absolute_depth_entry.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Azimuth (°):").grid(row=6, column=0, padx=10, pady=5)
azimuth_entry = tk.Entry(root)
azimuth_entry.grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Elevation (°):").grid(row=7, column=0, padx=10, pady=5)
elevation_entry = tk.Entry(root)
elevation_entry.grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="Region (ER, AMG, Other):").grid(row=8, column=0, padx=10, pady=5)
region_var = tk.StringVar(value="ER")  # Default selection

# Frame for radio buttons
region_frame = tk.Frame(root)
region_frame.grid(row=8, column=1, padx=10, pady=5)

region_er = tk.Radiobutton(region_frame, text="ER", variable=region_var, value="ER")
region_er.pack(side="left", padx=10)

region_amg = tk.Radiobutton(region_frame, text="AMG", variable=region_var, value="AMG")
region_amg.pack(side="left", padx=10)

region_other = tk.Radiobutton(region_frame, text="Other", variable=region_var, value="Other")
region_other.pack(side="left", padx=10)

# Dropdown menus for valid channels
tk.Label(root, text="Channel Categories:").grid(row=9, column=0, padx=10, pady=5)
channel_frame = tk.Frame(root)
channel_frame.grid(row=9, column=1, padx=10, pady=5)

channel_vars = {}
categories = ["Multiunits", "Well-isolated", "Disabled", "No spikes"]

for ch in channel_order:
    row_frame = tk.Frame(channel_frame)
    row_frame.pack(fill="x", pady=2)
    tk.Label(row_frame, text=f"Channel {ch}:").pack(side="left")
    channel_vars[ch] = tk.StringVar(value="No spikes")
    ttk.Combobox(row_frame, textvariable=channel_vars[ch], values=categories, state="readonly", width=15).pack(
        side="right")

# Buttons for saving and resetting
save_button = tk.Button(root, text="Save", command=save_to_csv)
save_button.grid(row=10, column=0, pady=10)

reset_button = tk.Button(root, text="Reset to Defaults", command=reset_to_defaults)
reset_button.grid(row=10, column=1, pady=10)

# Run the application
root.mainloop()
