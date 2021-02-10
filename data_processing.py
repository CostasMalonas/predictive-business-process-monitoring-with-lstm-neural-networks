import pandas as pd
import pm4py

"""
1ος τρόπος: Πάιρνω από το EVENTS_DATASET το action και το βάζω στο myfile.txt.
Όταν αλλάζω trace αλλάζω γραμμή.
"""
df = pd.read_csv('C:/Users/user/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/ΔΙΠΛΩΜΑΤΙΚΗ_2/EVENTS_DATASET.csv')
df.head()
df_2 = df[['Action', 'Org_Resource']]
df_2.head()


data = df_2.values.tolist()
data[0][0]


file1 = open("C:/Users/user/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/ΔΙΠΛΩΜΑΤΙΚΗ_2/myfile.txt", "w") 
i = 0
j = 0
user = data[0][1]

while i < 2000:
    while user == data[i][1]:
        file1.write(data[i][0] + ' ')
        i += 1
    user = data[i][1]
    file1.write('\n')
    #i += 1
file1.close()


"""
2ος τρόπος: Διαβάζω το xes file με την βιβλιοθήκη pm4py και βάζω τα ακρωνύμια
των events στο log_acron.txt. Παίρνω το concept:name που είναι πιο σωστό.
"""

log = pm4py.read_xes(r'C:\Users\user\Desktop\ΔΙΠΛΩΜΑΤΙΚΗ\12696884\BPI Challenge 2017.xes\BPI Challenge 2017.xes')
log[2000][2]


file1 = open("C:/Users/user/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/ΔΙΠΛΩΜΑΤΙΚΗ_2/log_Acron.txt", "w") 
for trace in log:
    for event in trace:
       file1.write(event['concept:name'] + ' ')
    file1.write('\n')   
file1.close()