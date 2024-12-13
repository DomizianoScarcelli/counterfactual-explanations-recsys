from typing import Optional

import pandas as pd
import zmq


# TODO: this is cool, but not correct yet. Write_pipe needs a pipe to write, which is created when the process is terminated, and so will never be ready.
class PipeIO:
    def __init__(self, address: str = "tcp://127.0.0.1:5555"):
        """
        Initializes the ZMQ pipe IO utility.

        :param address: The ZMQ address to bind or connect to.
        """
        self.address = address

    def write_pipe(self, df: pd.DataFrame):
        """
        Sends a DataFrame over a ZMQ socket to the specified address.
        Ensures the socket is ready before sending data.
        """
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)  # Use PUSH socket to send data
        socket.bind(self.address)

        # Poll the socket to ensure it is ready to send data
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLOUT)
        sockets = dict(
            poller.poll(1000)
        )  # Wait for 1 second for the socket to be ready

        if socket in sockets and sockets[socket] == zmq.POLLOUT:
            # Serialize the DataFrame to JSON
            df_json = df.to_json(orient="split")
            assert df_json  # Ensure the DataFrame was serialized
            socket.send_string(df_json)  # Send serialized DataFrame
            print(f"DataFrame sent to {self.address}.")
        else:
            print("Socket not ready to send data.")

        socket.close()
        context.term()

    def read_pipe(self) -> Optional[pd.DataFrame]:
        """
        Receives a DataFrame over a ZMQ socket from the specified address.
        Uses polling to ensure non-blocking behavior.
        :return: A pandas DataFrame or None if no data is received.
        """
        context = zmq.Context()
        socket = context.socket(zmq.PULL)  # Use PULL socket to receive data
        socket.connect(self.address)

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)  # Monitor the socket for incoming data

        try:
            # Poll for messages with a timeout (e.g., 5000 ms)
            events = dict(poller.poll(timeout=1000))  # Adjust timeout as needed

            if socket in events:  # Check if the socket has data
                df_json = socket.recv_string()  # Receive serialized DataFrame
                df = pd.read_json(df_json, orient="split")  # Deserialize to DataFrame
                print(f"DataFrame received from {self.address}.")
                return df
            else:
                print("No data available to read.")
                return None
        except zmq.ZMQError as e:
            print(f"Error receiving data: {e}")
            return None
        finally:
            poller.unregister(socket)  # Unregister the socket from the poller
            socket.close()
            context.term()
