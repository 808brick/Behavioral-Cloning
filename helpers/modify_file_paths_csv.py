import csv
from collections import deque

file_path = '/opt/Behavioral-Cloning/data/driving_log.csv'
from_str = "IMG/"
to_str = "/opt/Behavioral-Cloning/data/IMG/"

csv_data = deque()

with open(file_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    print("Reading in CSV data ...", end="")
    for row in csv_reader:
        center_image_path = row[0].replace(from_str, to_str)
        left_image_path = row[1].replace(from_str, to_str)
        right_image_path = row[2].replace(from_str, to_str)

        csv_data.append([center_image_path, left_image_path, right_image_path, row[3], row[4], row[5], row[6] ])
        

    print(" Done")


# print(csv_data[0])

print("Modifying CSV data ... ", end="")

with open(file_path, mode="w") as csv_file:
	csv_writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

	for row in csv_data:
		# print(row)
		csv_writer.writerow(row)

	print(" Done")