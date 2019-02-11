# importing csv module
import csv

# csv file name
filename = "Datasets_PRML_A1/Dataset_1_Team_39.csv"

# initializing the titles and rows list
fields = []
rows = []

# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

new_rows = []
for row in rows:
    temp = []
    for c in row:
        temp.append(float(c))
    new_rows.append(temp)

n_rows = len(rows)
n_cols = len(rows[0])

# number of features
n_feat = n_cols - 1

# number of class labels
m = 0
m_max = 0

if n_cols > 1:
    for r in new_rows:
        if r[n_cols-1] > m:
            m_max = r[n_cols-1]

m = int(m_max + 1)
classlabels = [0] * m

print("Number of rows = %d" %(n_rows))
print("Number of columns = %d" %(n_cols))
print("Number of features = %d" %(n_feat))
print("Number of class labels = %d (0 to %d)" %(m, m_max))

# finding prior probabilities
for row in new_rows:
    classlabels[int(row[n_cols-1])] = classlabels[int(row[n_cols-1])] + 1

priors = [x / n_rows for x in classlabels]

print(priors)
