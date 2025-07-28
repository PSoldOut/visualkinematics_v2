import re

def parse_cpu_lines(filename):
    

    # Regulärer Ausdruck zum Extrahieren der Werte
    pattern = r'%CPU\(s\):\s+([\d,]+)\s+us,\s+([\d,]+)\s+sy,\s+([\d,]+)\s+ni,\s+([\d,]+)\s+id'

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Kommazahlen in Punktzahlen umwandeln
                us = float(match.group(1).replace(',', '.'))
                sy = float(match.group(2).replace(',', '.'))
                ni = float(match.group(3).replace(',', '.'))
                id = float(match.group(4).replace(',', '.'))
                
                total = us + sy + ni + id
                cpu_sums.append(total)
                cpu_sums_only_us.append(us)





def sum_res_memory(filename):
    total_res_kb = 0

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if 'jupyter+' in line or 'python' in line:
                parts = line.split()
                if len(parts) >= 6:
                    res_str = parts[5]  # RES steht in Spalte 6 (Index 5)
                    
                    # Einheiten-Handling (z. B. K, M, G)
                    try:
                        if res_str.lower().endswith('g'):
                            res_kb = float(res_str[:-1].replace(',', '.')) * 1024 * 1024
                        elif res_str.lower().endswith('m'):
                            res_kb = float(res_str[:-1].replace(',', '.')) * 1024
                        elif res_str.lower().endswith('k'):
                            res_kb = float(res_str[:-1].replace(',', '.'))
                        else:
                            res_kb = float(res_str.replace(',', '.'))  # Falls keine Einheit
                        
                        total_res_kb += res_kb
                    except ValueError:
                        continue  # Falls etwas nicht konvertierbar ist, überspringen

    # Ausgabe als MiB
    total_res_mib = total_res_kb / 1024
    return total_res_mib


# Beispielnutzung
if __name__ == '__main__':
    dateiname = '../../perfTest/top_out4clients_ohne_restsessions.txt'  # deine Datei hier eintragen
    cpu_sums = []
    cpu_sums_only_us = []
    parse_cpu_lines(dateiname)
    print(f"durchschnittliche CPU-Auslastung in {sum(cpu_sums)/len(cpu_sums)}%")
    print(f"davon userspace {sum(cpu_sums_only_us)/len(cpu_sums_only_us)}%")

    res_mib = sum_res_memory(dateiname)
    print(f"durchschnittliche RES-Speichernutzung (jupyter+ und python): {(res_mib/len(cpu_sums)):.2f} MiB")