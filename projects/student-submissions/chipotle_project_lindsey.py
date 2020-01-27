
import csv
from collections import namedtuple   # Convenient to store the data rows

DATA_FILE = r'C:\Users\linds\.spyder-py3\chipotle.tsv'

# Specify that the delimiter is a tab character
with open(DATA_FILE, mode='r') as f:
    chipotle_csv = [row for row in csv.reader(f, delimiter='\t')]
"""Part 2: Separate file_nested_list into the header and the data."""
header = chipotle_csv[0]
header
ChipotleData = namedtuple("ChipotleData", header)
data = [ChipotleData._make(x) for x in chipotle_csv[1:]]

"""Part 3: Calculate the average price of an order."""
# Count the number of unique order_ids
# Note: You could assume this is 1,834 as that's the maximum order_id, but it's best to check
num_orders = len(set(row.order_id for row in data))
num_orders
# We must convert the price from a string to a float
# Strip the dollar sign and trailing space
print(data[0].item_price)
print(data[0].item_price[1:-1])  
#2 strip the dollar sign and trailing space
print(data[0].item_price.strip(' $'))
# Create a list of prices
prices = [float(row.item_price.strip(' $')) for row in data]
# Calculate the average price of an order
print(sum(prices) / num_orders)
print(float(round(sum(prices) / num_orders,2)))   
  
"""Part 4: Create a list (or set) named unique_sodas containing all of unique sodas """
"""and soft drinks that Chipotle sells"""
# For each canned drink, get the choice description
#  (Note this assumes there is exactly one descr in the brackets!)
sodas = []
for row in data:
    if 'Canned' in row.item_name:
        sodas.append(row.choice_description.strip('[]'))
print(set(sodas))
#2
# Equivalent list comprehension (using an 'if' condition)
sodas_list = [row.choice_description.strip('[]')\
              for row in data if 'Canned' in row.item_name]
print(set(sodas_list))

"""Part 5: Calculate the average number of toppings per burrito."""

burrito_count = 0
topping_count = 0
# Calculate the number of toppings by counting the commas and adding 1
# Note: x += 1 is equivalent to x = x + 1
for row in data:
    if 'Burrito' in row.item_name:
        burrito_count += 1
        topping_count += (row.choice_description.count(',') + 1)
print(burrito_count)
print(topping_count)

avg_topping = topping_count/burrito_count
print(float(round(avg_topping,2)))

"""Part 6: Create a dictionary. Let the keys represent chip orders"""
"""and the values represent the total number of orders"""
#empty dictionary
chips = {}
# If chip order is already in a dictionary, then update the value for that key
for row in data:
    if 'Chips' in row.item_name:
        if row.item_name not in chips:
            chips[row.item_name]= int(row.quantity)
        else:
            chips[row.item_name] += int(row.quantity)
chips
#2
#defaultdictsaves you the trouble of checking whether or not a key already exists
from collections import defaultdict
chips_dic = defaultdict(int)
for row in data:
    if 'Chips' in row.item_name:
        chips_dic[row.item_name] += int(row.quantity)
chips_dic



































