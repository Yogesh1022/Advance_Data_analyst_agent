# Prompts related to data visualization tasks
PLOT_INSTRUCTION_PROMPT = """
The user wants to generate a {plot_type} chart.
The available columns in the dataset are: {data_columns}.
The user's query was: "{user_query}"

Based on the user's query and the available columns, identify the most suitable X and Y columns for a {plot_type} chart.
If the chart type is 'histogram', only provide the X column.
If the chart type is 'pie', provide 'names' (x_col) and 'values' (y_col).
Provide your answer in a JSON format with keys "x", "y" (if applicable), and "title".
Example for bar chart: {{"x": "CategoryColumn", "y": "ValueColumn", "title": "Sales by Category"}}
Example for histogram: {{"x": "NumericalColumn", "title": "Distribution of NumericalColumn"}}
Example for pie chart: {{"x": "CategoryColumn", "y": "ValueColumn", "title": "Proportion of Categories"}}
If you cannot determine suitable columns, return an empty JSON object {{}}.
"""