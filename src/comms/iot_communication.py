"""IoT communication module for traffic monitoring system."""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import paho.mqtt.client as mqtt
import websockets
from websockets.server import WebSocketServerProtocol


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MQTTTrafficPublisher:
    """MQTT publisher for traffic sensor data.
    
    This class publishes traffic sensor data to MQTT topics
    for real-time monitoring and analysis.
    """
    
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: str = "traffic_publisher",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """Initialize the MQTT publisher.
        
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            client_id: MQTT client ID
            username: MQTT username (optional)
            password: MQTT password (optional)
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        
        # Create MQTT client
        self.client = mqtt.Client(client_id=client_id)
        
        if username and password:
            self.client.username_pw_set(username, password)
        
        # Set up callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
        self.connected = False
        logger.info(f"Initialized MQTT publisher for {broker_host}:{broker_port}")
    
    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Dict, rc: int) -> None:
        """Handle MQTT connection."""
        if rc == 0:
            self.connected = True
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        """Handle MQTT disconnection."""
        self.connected = False
        logger.info("Disconnected from MQTT broker")
    
    def _on_publish(self, client: mqtt.Client, userdata: Any, mid: int) -> None:
        """Handle successful message publish."""
        logger.debug(f"Message published: {mid}")
    
    def connect(self) -> bool:
        """Connect to MQTT broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 5
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            return self.connected
            
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Disconnected from MQTT broker")
    
    def publish_sensor_data(
        self,
        vehicle_count: int,
        avg_speed: float,
        weather: int,
        hour: int,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Publish traffic sensor data.
        
        Args:
            vehicle_count: Number of vehicles per minute
            avg_speed: Average speed in km/h
            weather: Weather condition (0=clear, 1=rain)
            hour: Hour of day (0-23)
            timestamp: Unix timestamp (optional)
            
        Returns:
            True if publish successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        if timestamp is None:
            timestamp = time.time()
        
        # Create sensor data payload
        sensor_data = {
            "timestamp": timestamp,
            "vehicle_count": vehicle_count,
            "avg_speed": avg_speed,
            "weather": weather,
            "hour": hour,
        }
        
        # Publish to sensor topics
        topics = [
            "traffic/sensors/vehicle_count",
            "traffic/sensors/speed",
            "traffic/sensors/weather",
            "traffic/sensors/hour",
        ]
        
        values = [vehicle_count, avg_speed, weather, hour]
        
        success = True
        for topic, value in zip(topics, values):
            payload = json.dumps({"timestamp": timestamp, "value": value})
            result = self.client.publish(topic, payload, qos=1)
            
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.error(f"Failed to publish to {topic}")
                success = False
        
        # Publish combined sensor data
        combined_topic = "traffic/sensors/combined"
        combined_payload = json.dumps(sensor_data)
        result = self.client.publish(combined_topic, combined_payload, qos=1)
        
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error(f"Failed to publish to {combined_topic}")
            success = False
        
        return success
    
    def publish_prediction(
        self,
        prediction: int,
        confidence: float,
        features: List[float],
        timestamp: Optional[float] = None,
    ) -> bool:
        """Publish traffic congestion prediction.
        
        Args:
            prediction: Predicted congestion (0=smooth, 1=congested)
            confidence: Prediction confidence (0.0 to 1.0)
            features: Input features used for prediction
            timestamp: Unix timestamp (optional)
            
        Returns:
            True if publish successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        if timestamp is None:
            timestamp = time.time()
        
        # Create prediction payload
        prediction_data = {
            "timestamp": timestamp,
            "prediction": prediction,
            "confidence": confidence,
            "features": features,
            "status": "congested" if prediction == 1 else "smooth",
        }
        
        # Publish to prediction topic
        topic = "traffic/predictions/congestion"
        payload = json.dumps(prediction_data)
        result = self.client.publish(topic, payload, qos=1)
        
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error(f"Failed to publish prediction to {topic}")
            return False
        
        logger.info(f"Published prediction: {prediction_data['status']} "
                   f"(confidence: {confidence:.3f})")
        
        return True
    
    def publish_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "info",
        timestamp: Optional[float] = None,
    ) -> bool:
        """Publish system alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (info, warning, error)
            timestamp: Unix timestamp (optional)
            
        Returns:
            True if publish successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        if timestamp is None:
            timestamp = time.time()
        
        # Create alert payload
        alert_data = {
            "timestamp": timestamp,
            "type": alert_type,
            "message": message,
            "severity": severity,
        }
        
        # Publish to alerts topic
        topic = "traffic/alerts"
        payload = json.dumps(alert_data)
        result = self.client.publish(topic, payload, qos=2)
        
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error(f"Failed to publish alert to {topic}")
            return False
        
        logger.info(f"Published alert: {alert_type} - {message}")
        
        return True


class MQTTTrafficSubscriber:
    """MQTT subscriber for traffic monitoring data.
    
    This class subscribes to MQTT topics and processes incoming
    traffic sensor data and predictions.
    """
    
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: str = "traffic_subscriber",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """Initialize the MQTT subscriber.
        
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            client_id: MQTT client ID
            username: MQTT username (optional)
            password: MQTT password (optional)
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        
        # Create MQTT client
        self.client = mqtt.Client(client_id=client_id)
        
        if username and password:
            self.client.username_pw_set(username, password)
        
        # Set up callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        self.connected = False
        self.message_handlers: Dict[str, Callable] = {}
        
        logger.info(f"Initialized MQTT subscriber for {broker_host}:{broker_port}")
    
    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: Dict, rc: int) -> None:
        """Handle MQTT connection."""
        if rc == 0:
            self.connected = True
            logger.info("Connected to MQTT broker")
            
            # Subscribe to traffic topics
            topics = [
                "traffic/sensors/+",
                "traffic/predictions/+",
                "traffic/alerts",
            ]
            
            for topic in topics:
                result = self.client.subscribe(topic, qos=1)
                if result[0] == mqtt.MQTT_ERR_SUCCESS:
                    logger.info(f"Subscribed to {topic}")
                else:
                    logger.error(f"Failed to subscribe to {topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        """Handle MQTT disconnection."""
        self.connected = False
        logger.info("Disconnected from MQTT broker")
    
    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        """Handle incoming MQTT message."""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Call registered handler if available
            if topic in self.message_handlers:
                self.message_handlers[topic](topic, payload)
            else:
                logger.debug(f"Received message on {topic}: {payload}")
                
        except json.JSONDecodeError:
            logger.error(f"Failed to decode message on {topic}")
        except Exception as e:
            logger.error(f"Error processing message on {topic}: {e}")
    
    def register_handler(self, topic: str, handler: Callable) -> None:
        """Register a message handler for a topic.
        
        Args:
            topic: MQTT topic pattern
            handler: Function to handle messages
        """
        self.message_handlers[topic] = handler
        logger.info(f"Registered handler for {topic}")
    
    def connect(self) -> bool:
        """Connect to MQTT broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 5
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            return self.connected
            
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Disconnected from MQTT broker")


class WebSocketTrafficServer:
    """WebSocket server for real-time traffic monitoring.
    
    This class provides a WebSocket server for real-time
    traffic monitoring dashboard communication.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        """Initialize the WebSocket server.
        
        Args:
            host: Server hostname
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients: List[WebSocketServerProtocol] = []
        self.server = None
        
        logger.info(f"Initialized WebSocket server for {host}:{port}")
    
    async def register_client(self, websocket: WebSocketServerProtocol) -> None:
        """Register a new client connection."""
        self.clients.append(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
    
    async def unregister_client(self, websocket: WebSocketServerProtocol) -> None:
        """Unregister a client connection."""
        if websocket in self.clients:
            self.clients.remove(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle client connection."""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                # Echo back the message for now
                await websocket.send(f"Echo: {message}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def broadcast_traffic_data(self, data: Dict[str, Any]) -> None:
        """Broadcast traffic data to all connected clients.
        
        Args:
            data: Traffic data to broadcast
        """
        if not self.clients:
            return
        
        message = json.dumps(data)
        disconnected_clients = []
        
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            await self.unregister_client(client)
    
    async def start_server(self) -> None:
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.handle_client, self.host, self.port
        )
        logger.info(f"WebSocket server started on {self.host}:{self.port}")
    
    async def stop_server(self) -> None:
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")


class TrafficDataStreamer:
    """Stream traffic data through multiple communication channels.
    
    This class coordinates data streaming through MQTT and WebSocket
    for comprehensive traffic monitoring.
    """
    
    def __init__(
        self,
        mqtt_config: Optional[Dict[str, Any]] = None,
        websocket_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the traffic data streamer.
        
        Args:
            mqtt_config: MQTT configuration dictionary
            websocket_config: WebSocket configuration dictionary
        """
        # Default configurations
        mqtt_config = mqtt_config or {
            "broker_host": "localhost",
            "broker_port": 1883,
        }
        
        websocket_config = websocket_config or {
            "host": "localhost",
            "port": 8765,
        }
        
        # Initialize components
        self.mqtt_publisher = MQTTTrafficPublisher(**mqtt_config)
        self.websocket_server = WebSocketTrafficServer(**websocket_config)
        
        self.running = False
        
        logger.info("Initialized TrafficDataStreamer")
    
    async def start(self) -> None:
        """Start the data streaming services."""
        # Connect to MQTT broker
        if not self.mqtt_publisher.connect():
            logger.error("Failed to connect to MQTT broker")
            return
        
        # Start WebSocket server
        await self.websocket_server.start_server()
        
        self.running = True
        logger.info("Traffic data streaming started")
    
    async def stop(self) -> None:
        """Stop the data streaming services."""
        self.running = False
        
        # Disconnect from MQTT broker
        self.mqtt_publisher.disconnect()
        
        # Stop WebSocket server
        await self.websocket_server.stop_server()
        
        logger.info("Traffic data streaming stopped")
    
    async def stream_sensor_data(
        self,
        vehicle_count: int,
        avg_speed: float,
        weather: int,
        hour: int,
    ) -> None:
        """Stream sensor data through all channels.
        
        Args:
            vehicle_count: Number of vehicles per minute
            avg_speed: Average speed in km/h
            weather: Weather condition (0=clear, 1=rain)
            hour: Hour of day (0-23)
        """
        # Publish to MQTT
        self.mqtt_publisher.publish_sensor_data(
            vehicle_count, avg_speed, weather, hour
        )
        
        # Broadcast via WebSocket
        sensor_data = {
            "type": "sensor_data",
            "timestamp": time.time(),
            "vehicle_count": vehicle_count,
            "avg_speed": avg_speed,
            "weather": weather,
            "hour": hour,
        }
        
        await self.websocket_server.broadcast_traffic_data(sensor_data)
    
    async def stream_prediction(
        self,
        prediction: int,
        confidence: float,
        features: List[float],
    ) -> None:
        """Stream prediction data through all channels.
        
        Args:
            prediction: Predicted congestion (0=smooth, 1=congested)
            confidence: Prediction confidence (0.0 to 1.0)
            features: Input features used for prediction
        """
        # Publish to MQTT
        self.mqtt_publisher.publish_prediction(prediction, confidence, features)
        
        # Broadcast via WebSocket
        prediction_data = {
            "type": "prediction",
            "timestamp": time.time(),
            "prediction": prediction,
            "confidence": confidence,
            "features": features,
            "status": "congested" if prediction == 1 else "smooth",
        }
        
        await self.websocket_server.broadcast_traffic_data(prediction_data)
