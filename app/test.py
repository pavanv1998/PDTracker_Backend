def find_initial_sales(b, n, total_sales):
    """
    Solve for a in the sum of a geometric series S_n = a * (1 - b^n) / (1 - b)
    """
    return total_sales * (1 - b) / (1 - b**n)

total_sales = 75000
months = 26

# Guess a monthly growth rate
growth_rate = 1.10

# Find the initial sales value 'a' for the guessed growth rate
initial_sales = find_initial_sales(growth_rate, months, total_sales)

# Calculate the total to verify it's close to 200,000
calculated_total = initial_sales * (1 - growth_rate**months) / (1 - growth_rate)
print(f"Initial sales (a): {initial_sales:.2f}")
print(f"Calculated total sales: {calculated_total:.2f}")

sum = 0
# Print out the sales for each month
for month in range(1, months + 1):
    month_sales = initial_sales * growth_rate**(month - 1)
    sum += round(month_sales)
    print(f"Month {month}: {round(month_sales)}")

print("sum: " + str(sum))