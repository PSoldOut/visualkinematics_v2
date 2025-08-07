#sps -> steps per second
#die Bildrate im Browser blieb auf den Geräten unabhängig stabil bei 60 FPS
#die Aktualisierungsrate der Simulation scheint linear mit der Anzahl an Verbindungen kleiner zu werden und scheint insgesammt starken Schwankungen zu unterliegen

import matplotlib.pyplot as plt
import statistics

results:list[dict] = [
    {
        "connections" : 1,
        "port" : 8880,
        "fps" : 60,
        "sps" : [203, 181, 243, 238, 233, 199, 187, 212, 271, 200]
    },
    {
        "connections" : 4,
        "port" : 8880,
        "fps" : 60,
        "sps" : [73, 64, 58, 71, 67, 61, 165, 59, 171, 54]
    },
    {
        "connections" : 4,
        "port" : 8881,
        "fps" : 60,
        "sps" : [64, 127, 112, 48, 81, 77, 48, 69, 53, 77]
    },
    {
        "connections" : 4,
        "port" : 8882,
        "fps" : 60,
        "sps" : [110, 47, 56, 51, 122, 119, 169, 74, 101, 66]
    },
    {
        "connections" : 4,
        "port" : 8883,
        "fps" : 60,
        "sps" : [43, 64, 51, 40, 69, 67, 63, 91, 63, 105]
    },
    {
        "connections" : 8,
        "port" : 8880,
        "fps" : 60,
        "sps" : [39, 44, 51, 28, 99, 82, 63, 35, 65, 30]
    },
    {
        "connections" : 8,
        "port" : 8881,
        "fps" : 60,
        "sps" : [34, 32, 52, 30, 60, 34, 45, 30, 31, 30]
    },
    {
        "connections" : 8,
        "port" : 8882,
        "fps" : 60,
        "sps" : [38, 46, 33, 33, 31, 33, 27, 34, 33, 32]
    },
    {
        "connections" : 8,
        "port" : 8883,
        "fps" : 60,
        "sps" : [31, 30, 33, 39, 43, 30, 40, 33, 39, 37]
    },
    {
        "connections" : 8,
        "port" : 8884,
        "fps" : 60,
        "sps" : [30, 33, 34, 31, 32, 38, 32, 64, 84, 44]
    },
    {
        "connections" : 8,
        "port" : 8885,
        "fps" : 60,
        "sps" : [32, 34, 65, 42, 31, 32, 37, 36, 63, 33]
    },
    {
        "connections" : 8,
        "port" : 8886,
        "fps" : 60,
        "sps" : [47, 39, 30, 28, 29, 36, 40, 54, 40, 34]
    },
    {
        "connections" : 8,
        "port" : 8887,
        "fps" : 60,
        "sps" : [39, 52, 39, 34, 41, 83, 36, 59, 45, 31]
    },
]




connection_levels = [1, 4, 8]

# Daten vorbereiten
averages = []
std_devs = []

for conn in connection_levels:
    all_sps = []
    for result in results:
        if result["connections"] == conn:
            all_sps.extend(result["sps"])
    avg = statistics.mean(all_sps)
    std = statistics.stdev(all_sps)
    averages.append(avg)
    std_devs.append(std)

# Diagramm erstellen
plt.figure(figsize=(8, 5))
plt.bar(connection_levels, averages, yerr=std_devs, capsize=8)
plt.xlabel("Anzahl der Verbindungen")
plt.ylabel("SPS (Schritte pro Sekunde)")
plt.title("Durchschnittliche SPS mit Standardabweichung")
plt.xticks(connection_levels)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()





# Gruppiere SPS-Werte pro Verbindungsanzahl
connection_levels = [1, 4, 8]
sps_values_grouped = []

for conn in connection_levels:
    all_sps = []
    for result in results:
        if result["connections"] == conn:
            all_sps.extend(result["sps"])
    sps_values_grouped.append(all_sps)

# Boxplot erstellen
plt.figure(figsize=(8, 5))
plt.boxplot(
    sps_values_grouped,
    labels=connection_levels,
    showmeans=True,
    meanline=True,
    notch=False,
    flierprops=dict(marker="o", markerfacecolor="red", markersize=4),
    meanprops=dict(markerfacecolor="green", marker="D", markeredgecolor="green")
)
plt.xlabel("Anzahl der Verbindungen")
plt.ylabel("SPS (Schritte pro Sekunde)")
plt.title("Boxplot der SPS-Verteilung pro Verbindungsanzahl")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

print(averages)