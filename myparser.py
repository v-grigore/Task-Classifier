import os
import json
from scapy.all import rdpcap, TCP
import concurrent.futures


def parse_pcap_with_scapy(file_path, name, test):
    packets = rdpcap(file_path)
    payloads = []

    for packet in packets:
        if packet.haslayer(TCP) and packet[TCP].payload:
            payload = bytes(packet[TCP].payload).hex()
            payloads.append(payload)

    if payloads:
        combined_payload = "".join(payloads)
        return parse_payload(combined_payload, name, test)
    return []


def parse_payload(payloads, name, test):
    json_object = []

    try:
        data = bytes.fromhex(payloads)
        json_data = json.loads(data)
        json_data["filename"] = name
        if test:
            json_data["label"] = 0
        json_object.append(json_data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error decoding payload: {e}")

    return json_object


def parse_json(path, output, test, file_limit=50000):
    all_json_objects = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, file in enumerate(os.listdir(path)):
            if i >= file_limit:
                break
            file_path = os.path.join(path, file)
            futures.append(
                executor.submit(parse_pcap_with_scapy, file_path, file, test)
            )

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                all_json_objects.extend(result)

    if all_json_objects:
        with open(output, "w") as outfile:
            json.dump(all_json_objects, outfile, indent=4)
    else:
        print("No valid JSON objects found.")


parse_json("/usr/src/app/InputData/test", "test.json", True)
parse_json("/usr/src/app/InputData/train", "train.json", False)
