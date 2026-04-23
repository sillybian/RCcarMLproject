import socket

def send_udp_command(ip, port, command):
    try:
        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Send the command to the RC car
        sock.sendto(command.encode('utf-8'), (ip, port))
        print(f"Sent: {command} to {ip}:{port}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        sock.close()

if __name__ == "__main__":
    # Replace with the IP address of your RC car and the port it is listening on
    car_ip = "10.42.1.222"  # Replace with the IP address printed in the Serial Monitor
    car_port = 8888         # Port the RC car is listening on

    print("Enter commands to control the RC car (A: Forward, L: Left, R: Right, B: Back, E: Stop). Type 'exit' to quit.")
    while True:
        command = input("Command: ").strip().upper()
        if command == "EXIT":
            break
        elif command in ["A", "L", "R", "B", "E"]:
            send_udp_command(car_ip, car_port, command)
        else:
            print("Invalid command. Use A, L, R, B, or E.")