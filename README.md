# capstone_project                 

# overview

To develop a real-time streaming model that dynamically calculates parking prices based on occupancy levels,nearby traffic,special events,simulating live data ingestion and visualizing the pricing evolution over time.

# Technologies:

Pathway: Real-time data streaming and processing.

Pandas & NumPy: Data manipulation and preparation.

Bokeh : Interactive data visualization.

Python :  pricing function logic.


# Workflow Summary:
--> Data Preparation:

Load parking occupancy data from a CSV file.

Combine date and time columns to create a proper timestamp.

--> Streaming Simulation:

Use Pathway’s replay_csv to simulate real-time data streaming from the static CSV.

Control the ingestion rate to mimic real-world scenarios.

--> Feature Engineering:

Parse timestamps and extract daily aggregations to support time-based analysis.

--> Pricing Model:

Apply a  dynamic pricing model where price increases with occupancy rate,traffic near by , special days , type of vehicle

The pricing formula adapts in real-time based on parking demand.

--> Visualization:

Use Bokeh  to create interactive, real-time visualizations of parking prices.

The dashboard updates automatically as the stream progresses.

--> Pipeline Execution:

Start the Pathway pipeline to continuously process and visualize the streaming data in the background.


# architecture diagram 

![architecture diagram](https://github.com/user-attachments/assets/b9d8bffc-8bab-4fbc-acb2-d0396eac2301)















               
               
