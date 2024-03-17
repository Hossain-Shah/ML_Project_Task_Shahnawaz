# dependencies import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from itertools import combinations
from collections import Counter

data = pd.read_csv("data/Online_Retail_Data_Set.csv", encoding = "unicode_escape")
print(data)

# unit price's distribution
sns.set(style="whitegrid")
sns.boxplot(y = 'UnitPrice', data=data)
plt.savefig('outputs/unit_price_boxplot.png')  
plt.show()

data = data[data['UnitPrice']>0] #filtering our data
print("\nUnit-price threshold:", data['UnitPrice'].min()) # without debt

# quantity's distribution
sns.set(style="whitegrid")
sns.boxplot(y = 'Quantity', data=data)
plt.savefig('outputs/quantity_boxplot.png')  
plt.show()

data = data[data['Quantity']>0]
print("\nPurchased product threshold:", data['Quantity'].min())

# sales's distribution
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format="%d-%m-%Y %H:%M")
data['Hour']=data['InvoiceDate'].dt.hour
BestTimeAdds = data.groupby('Hour').count().reset_index()
plt.plot(BestTimeAdds['Hour'],BestTimeAdds['InvoiceNo']/1000)
plt.xlabel('Hour')
plt.ylabel('Sales')
plt.savefig('outputs/sales_boxplot.png')  
plt.grid()
formatter = StrMethodFormatter('{x:.0f}k')
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()

# country-wise customer distribution
custCountry = data['Country'].value_counts().reset_index(drop = False).head(10)
custCountry.columns = ['Country','Counts'] 
plt.figure(figsize = (20,10))
ax =sns.barplot(x= 'Country', y = 'Counts' , data = custCountry, palette = 'coolwarm')
ax.bar_label(ax.containers[0])
plt.title("Customer Distribution Country-Wise",fontsize=20)
plt.ylabel("Orders",fontsize = 16)
plt.xlabel('Country',fontsize = 16)
plt.savefig('outputs/region_sales.png')  
plt.xticks(fontsize = 14)
plt.yscale('log')
