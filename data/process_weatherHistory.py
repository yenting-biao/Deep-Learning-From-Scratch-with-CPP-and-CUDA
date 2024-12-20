import pandas as pd


def main():
    raw_data_path = "./raw_data/weatherHistory.csv"
    output_data_path = "./processed_data/weatherHistory.csv"

    data = pd.read_csv(raw_data_path)

    # sort the data by date
    data["Formatted Date"] = pd.to_datetime(data["Formatted Date"], utc=True)
    data = data.sort_values(by="Formatted Date")

    # Only keep the columns that we need
    data = data[
        [
            # "Formatted Date",
            "Temperature (C)",
            "Apparent Temperature (C)",
            "Humidity",
        ]
    ]

    # Drop other columns
    data = data.dropna()

    # Predict the apparent temperature given the humidity of that day, and the humidity, temperature, and apparent temperature of the previous three days.
    data["Temperature (C) - 1"] = data["Temperature (C)"].shift(1)
    data["Temperature (C) - 2"] = data["Temperature (C)"].shift(2)
    data["Temperature (C) - 3"] = data["Temperature (C)"].shift(3)
    data["Apparent Temperature (C) - 1"] = data["Apparent Temperature (C)"].shift(1)
    data["Apparent Temperature (C) - 2"] = data["Apparent Temperature (C)"].shift(2)
    data["Apparent Temperature (C) - 3"] = data["Apparent Temperature (C)"].shift(3)
    data["Humidity - 1"] = data["Humidity"].shift(1)
    data["Humidity - 2"] = data["Humidity"].shift(2)
    data["Humidity - 3"] = data["Humidity"].shift(3)

    # drop the first three rows
    data = data.dropna()

    # Save the processed data
    data.to_csv(output_data_path, index=False)


if __name__ == "__main__":
    main()
