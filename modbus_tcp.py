import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty

import pymodbus.client.tcp
from pymodbus.exceptions import ModbusException, ConnectionException


class ModbusDataType(Enum):
    """Enumeration for Modbus data types."""
    INT16 = "int16"
    INT32 = "int32"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    BOOLEAN = "boolean"


@dataclass
class ModbusRegister:
    """Data class for Modbus register configuration."""
    address: int
    data_type: ModbusDataType
    name: str
    description: str
    unit: str = ""
    scale_factor: float = 1.0
    byte_order: str = "big"
    validation_rules: Optional[Dict[str, Any]] = None


class SimpleLogger:
    """Simple logger for Modbus TCP reader."""
    
    def __init__(self, log_level: int = 0):
        """
        Initialize logger.
        
        Args:
            log_level: Log level (0=info, 1=warning, 2=error)
        """
        self.log_level = log_level
    
    def log(self, data: Any, log_type: int = 0, visibility: str = "TD", tag: str = "ModbusTCPReader") -> None:
        """
        Log a message.
        
        Args:
            data: Data to log
            log_type: Type of log (0=info, 1=warning, 2=error)
            visibility: Visibility level
            tag: Tag for the log
        """
        if log_type >= self.log_level:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            level_str = {0: "INFO", 1: "WARNING", 2: "ERROR"}.get(log_type, "INFO")
            print(f"[{timestamp}] [{level_str}] [{tag}] {data}")


class ConnectionPool:
    """
    Connection pool for Modbus TCP clients.
    Manages multiple connections to improve performance and reliability.
    """
    
    def __init__(self, max_connections: int = 5, max_idle_time: float = 300.0):
        """
        Initialize connection pool.
        
        Args:
            max_connections: Maximum number of connections in pool
            max_idle_time: Maximum idle time for connections in seconds
        """
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.connections: Queue = Queue(maxsize=max_connections)
        self.active_connections = 0
        self.lock = threading.Lock()
        
    def get_connection(self, host: str, port: int, **kwargs) -> Optional[pymodbus.client.tcp.ModbusTcpClient]:
        """
        Get a connection from the pool or create a new one.
        
        Args:
            host: TCP host address
            port: TCP port
            **kwargs: Additional connection parameters
            
        Returns:
            ModbusTcpClient instance or None if failed
        """
        try:
            # Try to get existing connection from pool
            connection = self.connections.get_nowait()
            if self._is_connection_valid(connection):
                return connection
            else:
                self._close_connection(connection)
        except Empty:
            pass
        
        # Create new connection if pool is not full
        with self.lock:
            if self.active_connections < self.max_connections:
                connection = self._create_connection(host, port, **kwargs)
                if connection:
                    self.active_connections += 1
                return connection
        
        return None
    
    def return_connection(self, connection: pymodbus.client.tcp.ModbusTcpClient) -> None:
        """
        Return a connection to the pool.
        
        Args:
            connection: ModbusTcpClient instance to return
        """
        if connection and self._is_connection_valid(connection):
            try:
                self.connections.put_nowait(connection)
            except:
                # Pool is full, close the connection
                self._close_connection(connection)
        else:
            self._close_connection(connection)
    
    def _create_connection(self, host: str, port: int, **kwargs) -> Optional[pymodbus.client.tcp.ModbusTcpClient]:
        """
        Create a new TCP connection.
        
        Args:
            host: TCP host address
            port: TCP port
            **kwargs: Additional connection parameters
            
        Returns:
            ModbusTcpClient instance or None if failed
        """
        try:
            client = pymodbus.client.tcp.ModbusTcpClient(
                host=host,
                port=port,
                **kwargs
            )
            
            if client.connect():
                return client
            else:
                client.close()
                return None
        except Exception:
            return None
    
    def _is_connection_valid(self, connection: pymodbus.client.tcp.ModbusTcpClient) -> bool:
        """
        Check if a connection is still valid.
        
        Args:
            connection: ModbusTcpClient instance to check
            
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            return connection.is_socket_open()
        except:
            return False
    
    def _close_connection(self, connection: pymodbus.client.tcp.ModbusTcpClient) -> None:
        """
        Close a connection.
        
        Args:
            connection: ModbusTcpClient instance to close
        """
        try:
            if connection:
                connection.close()
        except:
            pass
        
        with self.lock:
            self.active_connections = max(0, self.active_connections - 1)
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        while not self.connections.empty():
            try:
                connection = self.connections.get_nowait()
                self._close_connection(connection)
            except Empty:
                break


class ModbusTCPReader:
    """
    OOP wrapper for Modbus TCP/IP communication.
    
    This class provides a clean, object-oriented interface for reading data
    from Modbus TCP devices with connection pooling, health monitoring,
    and automatic retry mechanisms.
    """
    
    def __init__(
        self,
        device_id: str,
        host: str,
        port: int = 502,
        timeout: float = 1.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        max_connections: int = 5,
        health_check_interval: float = 30.0,
        logger: Optional[SimpleLogger] = None
    ):
        """
        Initialize Modbus TCP Reader.
        
        Args:
            device_id: Unique identifier for the device
            host: TCP host address
            port: TCP port (default: 502)
            timeout: Read timeout in seconds
            retry_count: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            max_connections: Maximum number of connections in pool
            health_check_interval: Health check interval in seconds
            logger: Logger instance
        """
        self.device_id = device_id
        self.device_type = "modbus_tcp"
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.health_check_interval = health_check_interval
        
        self.logger = logger or SimpleLogger()
        self.connection_pool = ConnectionPool(max_connections)
        self.last_read_time: Optional[float] = None
        self.last_health_check: Optional[float] = None
        self.health_status = "unknown"
        
        # Register configuration
        self.registers: Dict[str, ModbusRegister] = {}
        
        # Connection statistics
        self.stats = {
            "total_reads": 0,
            "successful_reads": 0,
            "failed_reads": 0,
            "connection_errors": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "last_error": None
        }
        
        # Health monitoring thread
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.health_monitor_running = False
        
        # Start health monitoring
        self._start_health_monitor()
    
    def add_register(self, register: ModbusRegister) -> None:
        """
        Add a register configuration to the reader.
        
        Args:
            register: ModbusRegister configuration
        """
        self.registers[register.name] = register
        self.logger.log(
            data=f"Added register: {register.name} at address {register.address}",
            log_type=0,
            visibility="TD",
            tag="ModbusTCPReader"
        )
    
    def add_registers(self, registers: List[ModbusRegister]) -> None:
        """
        Add multiple register configurations.
        
        Args:
            registers: List of ModbusRegister configurations
        """
        for register in registers:
            self.add_register(register)
    
    def read_register(self, register: ModbusRegister) -> Tuple[bool, Any]:
        """
        Read a single register from the device.
        
        Args:
            register: Register configuration to read
            
        Returns:
            Tuple of (success: bool, value: Any)
        """
        connection = None
        
        for attempt in range(self.retry_count):
            try:
                # Get connection from pool
                connection = self.connection_pool.get_connection(
                    host=self.host,
                    port=self.port,
                    timeout=self.timeout
                )
                
                if connection is None:
                    self.stats["pool_misses"] += 1
                    self.stats["connection_errors"] += 1
                    self.stats["last_error"] = "Failed to get connection from pool"
                    
                    if attempt < self.retry_count - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return False, None
                else:
                    self.stats["pool_hits"] += 1
                
                # Determine number of registers to read based on data type
                register_count = self._get_register_count(register.data_type)
                
                # Read registers
                result = connection.read_holding_registers(
                    address=register.address,
                    count=register_count
                )
                
                if result.isError():
                    raise ModbusException(f"Modbus error: {result}")
                
                # Transform raw data to appropriate type
                value = self._transform_data(result.registers, register)
                
                self.stats["successful_reads"] += 1
                self.last_read_time = time.time()
                
                return True, value
                
            except (ModbusException, ConnectionException) as e:
                self.stats["failed_reads"] += 1
                self.stats["last_error"] = str(e)
                
                if attempt < self.retry_count - 1:
                    self.logger.log(
                        data=f"Read attempt {attempt + 1} failed, retrying: {str(e)}",
                        log_type=1,
                        visibility="TD",
                        tag="ModbusTCPReader"
                    )
                    time.sleep(self.retry_delay)
                else:
                    self.logger.log(
                        data=f"All read attempts failed: {str(e)}",
                        log_type=2,
                        visibility="TD",
                        tag="ModbusTCPReader"
                    )
                    return False, None
            finally:
                # Return connection to pool
                if connection:
                    self.connection_pool.return_connection(connection)
        
        return False, None
    
    def read_registers(self, register_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Read multiple registers from the device.
        
        Args:
            register_names: List of register names to read. If None, reads all configured registers.
            
        Returns:
            Dictionary mapping register names to their values
        """
        if register_names is None:
            register_names = list(self.registers.keys())
        
        results = {}
        self.stats["total_reads"] += 1
        
        for register_name in register_names:
            if register_name not in self.registers:
                self.logger.log(
                    data=f"Register '{register_name}' not configured",
                    log_type=2,
                    visibility="TD",
                    tag="ModbusTCPReader"
                )
                continue
            
            register = self.registers[register_name]
            success, value = self.read_register(register)
            
            if success:
                results[register_name] = value
            else:
                results[register_name] = None
        
        return results
    
    def read_data(self) -> Dict[str, Any]:
        """
        Read data from the device.
        
        Returns:
            Dictionary containing device data
        """
        return self.read_registers()
    
    def save_data(self, data: Dict[str, Any]) -> bool:
        """
        Save data (placeholder method).
        
        Args:
            data: Data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # This would typically save to the database
            # For now, just log the data
            self.logger.log(
                data=data,
                log_type=0,
                visibility="TD",
                tag="ModbusTCPReader"
            )
            return True
        except Exception as e:
            self.logger.log(
                data=f"Failed to save data: {str(e)}",
                log_type=2,
                visibility="TD",
                tag="ModbusTCPReader"
            )
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get device status information.
        
        Returns:
            Dictionary containing device status
        """
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "host": self.host,
            "port": self.port,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "health_check_interval": self.health_check_interval,
            "health_status": self.health_status,
            "last_read_time": self.last_read_time,
            "last_health_check": self.last_health_check,
            "register_count": len(self.registers),
            "pool_stats": {
                "active_connections": self.connection_pool.active_connections,
                "max_connections": self.connection_pool.max_connections,
                "pool_size": self.connection_pool.connections.qsize()
            },
            "stats": self.stats.copy()
        }
    
    def check_health(self) -> bool:
        """
        Check the health of the TCP connection.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to read a simple register to test connection
            test_register = ModbusRegister(
                address=0,
                data_type=ModbusDataType.INT16,
                name="health_check",
                description="Health check register"
            )
            
            success, _ = self.read_register(test_register)
            
            if success:
                self.health_status = "healthy"
                self.last_health_check = time.time()
                return True
            else:
                self.health_status = "unhealthy"
                self.last_health_check = time.time()
                return False
                
        except Exception as e:
            self.health_status = "error"
            self.last_health_check = time.time()
            self.logger.log(
                data=f"Health check failed: {str(e)}",
                log_type=2,
                visibility="TD",
                tag="ModbusTCPReader"
            )
            return False
    
    def _start_health_monitor(self) -> None:
        """Start the health monitoring thread."""
        if not self.health_monitor_running:
            self.health_monitor_running = True
            self.health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True
            )
            self.health_monitor_thread.start()
    
    def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while self.health_monitor_running:
            try:
                self.check_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.log(
                    data=f"Health monitor error: {str(e)}",
                    log_type=2,
                    visibility="TD",
                    tag="ModbusTCPReader"
                )
                time.sleep(self.health_check_interval)
    
    def _get_register_count(self, data_type: ModbusDataType) -> int:
        """
        Get the number of registers needed for a data type.
        
        Args:
            data_type: Modbus data type
            
        Returns:
            Number of registers needed
        """
        register_counts = {
            ModbusDataType.INT16: 1,
            ModbusDataType.INT32: 2,
            ModbusDataType.FLOAT32: 2,
            ModbusDataType.FLOAT64: 4,
            ModbusDataType.STRING: 1,  # Default for string
            ModbusDataType.BOOLEAN: 1
        }
        return register_counts.get(data_type, 1)
    
    def _transform_data(self, registers: List[int], register_config: ModbusRegister) -> Any:
        """
        Transform raw register data to appropriate data type.
        
        Args:
            registers: Raw register values
            register_config: Register configuration
            
        Returns:
            Transformed data value
        """
        try:
            if register_config.data_type == ModbusDataType.INT16:
                value = registers[0]
            elif register_config.data_type == ModbusDataType.INT32:
                value = (registers[0] << 16) | registers[1]
            elif register_config.data_type == ModbusDataType.FLOAT32:
                import struct
                # Convert registers to bytes
                bytes_data = b""
                for reg in registers:
                    bytes_data += reg.to_bytes(2, byteorder=register_config.byte_order)
                value = struct.unpack('>f', bytes_data)[0]
            elif register_config.data_type == ModbusDataType.FLOAT64:
                import struct
                bytes_data = b""
                for reg in registers:
                    bytes_data += reg.to_bytes(2, byteorder=register_config.byte_order)
                value = struct.unpack('>d', bytes_data)[0]
            elif register_config.data_type == ModbusDataType.STRING:
                # Convert registers to string
                bytes_data = b""
                for reg in registers:
                    bytes_data += reg.to_bytes(2, byteorder=register_config.byte_order)
                value = bytes_data.decode('utf-8').strip('\x00')
            elif register_config.data_type == ModbusDataType.BOOLEAN:
                value = bool(registers[0])
            else:
                value = registers[0]
            
            # Apply scale factor
            value = value * register_config.scale_factor
            
            return value
            
        except Exception as e:
            self.logger.log(
                data=f"Data transformation error: {str(e)}",
                log_type=2,
                visibility="TD",
                tag="ModbusTCPReader"
            )
            return None
    
    def close(self) -> None:
        """Close the TCP reader and clean up resources."""
        self.health_monitor_running = False
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5.0)
        
        self.connection_pool.close_all()
        
        self.logger.log(
            data=f"Closed Modbus TCP reader for {self.host}:{self.port}",
            log_type=0,
            visibility="TD",
            tag="ModbusTCPReader"
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage and backward compatibility
def create_modbus_tcp_reader(
    device_id: str,
    host: str,
    port: int = 502,
    **kwargs
) -> ModbusTCPReader:
    """
    Factory function to create a Modbus TCP reader.
    
    Args:
        device_id: Unique identifier for the device
        host: TCP host address
        port: TCP port
        **kwargs: Additional arguments
        
    Returns:
        Configured ModbusTCPReader instance
    """
    return ModbusTCPReader(device_id, host, port, **kwargs)


# Backward compatibility function
def read_modbus_tcp_data(
    device_id: str,
    host: str,
    port: int,
    registers: List[ModbusRegister],
    **kwargs
) -> Dict[str, Any]:
    """
    Read data from Modbus TCP device (backward compatibility function).
    
    Args:
        device_id: Device identifier
        host: TCP host address
        port: TCP port
        registers: List of registers to read
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of register values
    """
    reader = ModbusTCPReader(device_id, host, port, **kwargs)
    reader.add_registers(registers)
    
    with reader:
        return reader.read_registers()