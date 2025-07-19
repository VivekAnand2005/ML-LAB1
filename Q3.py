import pandas as pd
import statistics
import matplotlib.pyplot as plt

df_stock = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
df_stock['Day'] = df_stock['Date'].dt.day_name()

price_col = df_stock.columns[3]
chg_col = df_stock.columns[8]

prices = df_stock[price_col]
mean_all = statistics.mean(prices)
var_all = statistics.variance(prices)

mean_wed = df_stock[df_stock['Day'] == 'Wednesday'][price_col].mean()
mean_apr = df_stock[df_stock['Date'].dt.month == 4][price_col].mean()

chg = df_stock[chg_col]
loss_prob = len(chg[chg < 0]) / len(chg)
profit_wed = len(df_stock[(df_stock['Day'] == 'Wednesday') & (chg > 0)]) / len(df_stock[df_stock['Day'] == 'Wednesday'])

# Plot
plt.scatter(df_stock['Day'], chg)
plt.title("Chg% vs Day")
plt.ylabel("Change %")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
