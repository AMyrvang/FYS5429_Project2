import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn theme setup for consistent aesthetics
sns.set_theme(style="whitegrid", palette="colorblind")
cm = 1 / 2.54

# Define plot parameters suitable for LaTeX integration
plt.rcParams.update({
    "legend.fontsize": 6,       
    "axes.labelsize": 7,         
    "axes.titlesize": 8,         
    "xtick.labelsize": 6,        
    "ytick.labelsize": 6,        
    "font.size": 7,              
    "figure.figsize": (9 * cm, 12 * cm),  
    "lines.markersize": 2.0,     
    "lines.linewidth": 0.5,      
    "grid.linestyle": '--',      
    "grid.alpha": 0.6            
})

df = pd.read_csv('Data/iskoras_measurements.csv', index_col=0)

# Rename the columns for clarity
df = df.rename(columns={
    'T$_\mathrm{soil}$': 'soil_temperature',
    'VWC': 'soil_volumetric_water_content',
    'T$_\mathrm{air}$': 'air_temperature',
    'SW$_\mathrm{in}$': 'shortwave_incoming',
    'LW$_\mathrm{in}$': 'longwave_incoming',
    'VPD': 'vpd',
    'Albedo': 'albedo',
    'T$_\mathrm{surf}$': 'surface_temperature',
    'NDVI': 'NDVI',
    'FSCA': 'FSCA',
    'F$_\mathrm{total}^\mathrm{CO2}$': 'co2_flux_filtered',
    'F$_\mathrm{total}^\mathrm{CH4}$': 'ch4_flux_filtered',
    'w$_\mathrm{palsa}$': 'palsa',
    'w$_\mathrm{ponds}$': 'ponds',
    'w$_\mathrm{fen}$': 'fen'
})

# Convert CH4 flux to mmol
df['ch4_flux_filtered'] = df['ch4_flux_filtered'] / 1000

# Initial plot for CO2 and CH4 emissions
plt.figure(figsize=(9 * cm, 12 * cm), dpi=300)
row_numbers = range(len(df))

# CO2 emissions plot
plt.subplot(2, 1, 1)
plt.plot(row_numbers, df['co2_flux_filtered'], label='CO2 Emissions', color='#404040', linewidth=0.5)
plt.title('Carbon Dioxide (CO2) Emissions')
plt.xlabel('Row Number')
plt.ylabel('CO2 Flux')
plt.legend()

# CH4 emissions plot
plt.subplot(2, 1, 2)
plt.plot(row_numbers, df['ch4_flux_filtered'], label='CH4 Emissions', color='darkgreen', linewidth=0.5)
plt.title('Methane (CH4) Emissions')
plt.xlabel('Row Number')
plt.ylabel('CH4 Flux')
plt.legend()
plt.tight_layout()
plt.savefig("Figs/Emissions_Plot.png", dpi=300)
plt.show()


# Data preprocessing
df = df.dropna() 
df.index = pd.to_datetime(df.index) 

# Extract date and time from the datetime index
df['date'] = df.index.date
df['time'] = df.index.time

# Reorder columns to place date and time at the beginning
df = df[['date', 'time'] + [col for col in df.columns if col not in ['date', 'time']]]

# Save the processed data to a new CSV file
df.to_csv('Data/processed_iskoras_measurements.csv', index=False)
print("Data processed and saved to processed_iskoras_measurements.csv.")

# Plot the processed CO2 and CH4 emissions
plt.figure(figsize=(9 * cm, 12 * cm), dpi=300)
row_numbers = range(len(df))

# CO2 emissions plot
plt.subplot(2, 1, 1)
plt.plot(row_numbers, df['co2_flux_filtered'], label='CO2 Emissions', color='#404040', linewidth=0.5)
plt.title('Carbon Dioxide (CO2) Emissions')
plt.xlabel('Row Number')
plt.ylabel('CO2 Flux (filtered)')
plt.legend()

# CH4 emissions plot
plt.subplot(2, 1, 2)
plt.plot(row_numbers, df['ch4_flux_filtered'], label='CH4 Emissions', color='darkgreen', linewidth=0.5)
plt.title('Methane (CH4) Emissions')
plt.xlabel('Row Number')
plt.ylabel('CH4 Flux (filtered)')
plt.legend()
plt.tight_layout()
plt.savefig("Figs/Processed_emissions_Plot.png", dpi=300)
plt.show()
