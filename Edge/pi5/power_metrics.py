"""
PMIC ADC Monitor Script
Periodically runs 'vcgencmd pmic_read_adc' and parses the output.
"""

import subprocess
import re
import time
import json
import csv
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import sys


class PMICMonitor:
    def __init__(self, interval: int = 60, output_format: str = "json"):
        """
        Initialize the PMIC monitor.

        Args:
            interval: Seconds between readings
            output_format: Output format ('json', 'csv', 'print')
        """
        self.interval = interval
        self.output_format = output_format
        self.csv_file = None
        self.csv_writer = None
        self.csv_headers_written = False

    def run_command(self) -> Optional[str]:
        """
        Run the vcgencmd command and return the output.

        Returns:
            Command output as string, or None if failed
        """
        try:
            result = subprocess.run(
                ["vcgencmd", "pmic_read_adc"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return result.stdout
            else:
                print(f"Command failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("Command timed out")
            return None
        except Exception as e:
            print(f"Error running command: {e}")
            return None

    def parse_output(self, output: str) -> Dict:
        """
        Parse the vcgencmd output into a structured dictionary.

        Args:
            output: Raw command output

        Returns:
            Dictionary with parsed data
        """
        data = {"timestamp": datetime.now().isoformat(), "currents": {}, "voltages": {}}

        lines = output.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Parse current readings
            current_match = re.match(r"^\s*(\S+)\s+current\((\d+)\)=([0-9.]+)A$", line)
            if current_match:
                name, channel, value = current_match.groups()
                data["currents"][name] = {
                    "channel": int(channel),
                    "value": float(value),
                    "unit": "A",
                }
                continue

            # Parse voltage readings
            voltage_match = re.match(r"^\s*(\S+)\s+volt\((\d+)\)=([0-9.]+)V$", line)
            if voltage_match:
                name, channel, value = voltage_match.groups()
                data["voltages"][name] = {
                    "channel": int(channel),
                    "value": float(value),
                    "unit": "V",
                }
                continue

        return data

    def output_data(self, data: Dict):
        """
        Output the parsed data in the specified format.

        Args:
            data: Parsed data dictionary
        """
        if self.output_format == "json":
            print(json.dumps(data, indent=2))

        elif self.output_format == "csv":
            self.output_csv(data)

        elif self.output_format == "print":
            self.output_formatted(data)

    def output_formatted(self, data: Dict):
        """
        Output data in a human-readable format.

        Args:
            data: Parsed data dictionary
        """
        print(f"\n=== PMIC Reading at {data['timestamp']} ===")

        print("\nCurrents:")
        for name, info in sorted(data["currents"].items()):
            print(
                f"  {name:15} (ch{info['channel']:2d}): {info['value']:8.5f} {info['unit']}"
            )

        print("\nVoltages:")
        for name, info in sorted(data["voltages"].items()):
            print(
                f"  {name:15} (ch{info['channel']:2d}): {info['value']:8.5f} {info['unit']}"
            )

        print("-" * 50)

    def setup_csv(self, data: Dict):
        """
        Set up CSV output file and writer.

        Args:
            data: Sample data to determine headers
        """
        if self.csv_file is None:
            filename = f"pmic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.csv_file = open(filename, "w", newline="")

            # Create headers
            headers = ["timestamp"]
            for name in sorted(data["currents"].keys()):
                headers.append(f"{name}_current_A")
            for name in sorted(data["voltages"].keys()):
                headers.append(f"{name}_voltage_V")

            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=headers)
            self.csv_writer.writeheader()
            self.csv_headers_written = True

            print(f"CSV output file: {filename}")

    def output_csv(self, data: Dict):
        """
        Output data to CSV file.

        Args:
            data: Parsed data dictionary
        """
        if not self.csv_headers_written:
            self.setup_csv(data)

        # Flatten data for CSV
        row = {"timestamp": data["timestamp"]}

        for name, info in data["currents"].items():
            row[f"{name}_current_A"] = info["value"]

        for name, info in data["voltages"].items():
            row[f"{name}_voltage_V"] = info["value"]

        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def run(self):
        """
        Main monitoring loop.
        """
        print(
            f"Starting PMIC monitoring (interval: {self.interval}s, format: {self.output_format})"
        )
        print("Press Ctrl+C to stop")

        try:
            while True:
                output = self.run_command()

                if output:
                    data = self.parse_output(output)
                    self.output_data(data)
                else:
                    print("Failed to get PMIC data")

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\nStopping monitor...")

        finally:
            if self.csv_file:
                self.csv_file.close()


def main():
    parser = argparse.ArgumentParser(description="Monitor Raspberry Pi PMIC ADC values")
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=60,
        help="Interval between readings in seconds (default: 60)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "csv", "print"],
        default="print",
        help="Output format (default: print)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run once and exit (for testing)"
    )

    args = parser.parse_args()

    monitor = PMICMonitor(interval=args.interval, output_format=args.format)

    if args.test:
        # Test mode - run once
        output = monitor.run_command()
        if output:
            data = monitor.parse_output(output)
            monitor.output_data(data)
        else:
            print("Failed to get PMIC data")
            sys.exit(1)
    else:
        # Normal monitoring mode
        monitor.run()


if __name__ == "__main__":
    main()
