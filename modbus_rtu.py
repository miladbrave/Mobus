import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import pymodbus.client.serial
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
    unit: Optional[str] = None
    scale_factor: float = 1.0
    byte_order: str = "big"
    word_order: str = "big"


class SimpleLogger:
    """Simple logger for Modbus RTU reader."""
    
    def __init__(self, log_level: int = 0):
        """
        Initialize logger.
        
        Args:
            log_level: Log level (0=info, 1=warning, 2=error)
        """
        self.log_level = log_level
    
    def log(self, data: Any, log_type: int = 0, visibility: str = "TD", tag: str = "ModbusRTUReader") -> None:
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


class ModbusRTUReader:
    """
    OOP wrapper for Modbus RTU communication.
    
    This class provides a clean, object-oriented interface for reading data
    from Modbus RTU devices with connection management, error handling,
    and data transformation capabilities.
    """
    
    def __init__(
        self,
        device_id: str,
        port: str,
        baudrate: int = 9600,
        parity: str = "N",
        stopbits: int = 1,
        bytesize: int = 8,
        timeout: float = 1.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        logger: Optional[SimpleLogger] = None
    ):
        """
        Initialize Modbus RTU Reader.
        
        Args:
            device_id: Unique identifier for the device
            port: Serial port (e.g., '/dev/ttyUSB0' or 'COM1')
            baudrate: Baud rate for serial communication
            parity: Parity bit ('N', 'E', 'O')
            stopbits: Number of stop bits
            bytesize: Number of data bits
            timeout: Read timeout in seconds
            retry_count: Number of retry attempts on failure
            retry_delay: Delay between retries in seconds
            logger: Logger instance
        """
        self.device_id = device_id
        self.device_type = "modbus_rtu"
        self.port = port
        self.baudrate = baudrate
        self.parity = parity
        self.stopbits = stopbits
        self.bytesize = bytesize
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        self.logger = logger or SimpleLogger()
        self.client: Optional[pymodbus.client.serial.ModbusSerialClient] = None
        self.is_connected = False
        self.last_read_time: Optional[float] = None
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        
        # Register configuration
        self.registers: Dict[str, ModbusRegister] = {}
        
        # Connection statistics
        self.stats = {
            "total_reads": 0,
            "successful_reads": 0,
            "failed_reads": 0,
            "connection_errors": 0,
            "last_error": None
        }
    
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
            tag="ModbusRTUReader"
        )
    
    def add_registers(self, registers: List[ModbusRegister]) -> None:
        """
        Add multiple register configurations.
        
        Args:
            registers: List of ModbusRegister configurations
        """
        for register in registers:
            self.add_register(register)
    
    def connect(self) -> bool:
        """
        Establish connection to the Modbus RTU device.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.client and self.is_connected:
                return True
            
            self.client = pymodbus.client.serial.ModbusSerialClient(
                port=self.port,
                baudrate=self.baudrate,
                parity=self.parity,
                stopbits=self.stopbits,
                bytesize=self.bytesize,
                timeout=self.timeout
            )
            
            if self.client.connect():
                self.is_connected = True
                self.connection_attempts = 0
                self.logger.log(
                    data=f"Connected to Modbus RTU device on {self.port}",
                    log_type=0,
                    visibility="TD",
                    tag="ModbusRTUReader"
                )
                return True
            else:
                self.is_connected = False
                self.connection_attempts += 1
                self.stats["connection_errors"] += 1
                self.stats["last_error"] = "Failed to connect"
                
                self.logger.log(
                    data=f"Failed to connect to Modbus RTU device on {self.port}",
                    log_type=2,
                    visibility="TD",
                    tag="ModbusRTUReader"
                )
                return False
                
        except Exception as e:
            self.is_connected = False
            self.connection_attempts += 1
            self.stats["connection_errors"] += 1
            self.stats["last_error"] = str(e)
            
            self.logger.log(
                data=f"Connection error: {str(e)}",
                log_type=2,
                visibility="TD",
                tag="ModbusRTUReader"
            )
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the Modbus RTU device."""
        if self.client and self.is_connected:
            self.client.close()
            self.is_connected = False
            self.logger.log(
                data=f"Disconnected from Modbus RTU device on {self.port}",
                log_type=0,
                visibility="TD",
                tag="ModbusRTUReader"
            )
    
    def read_register(self, register: ModbusRegister) -> Tuple[bool, Any]:
        """
        Read a single register from the device.
        
        Args:
            register: Register configuration to read
            
        Returns:
            Tuple of (success: bool, value: Any)
        """
        if not self.is_connected and not self.connect():
            return False, None
        
        for attempt in range(self.retry_count):
            try:
                # Determine number of registers to read based on data type
                register_count = self._get_register_count(register.data_type)
                
                # Read registers
                result = self.client.read_holding_registers(
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
                        tag="ModbusRTUReader"
                    )
                    time.sleep(self.retry_delay)
                    # Try to reconnect
                    self.disconnect()
                    self.connect()
                else:
                    self.logger.log(
                        data=f"All read attempts failed: {str(e)}",
                        log_type=2,
                        visibility="TD",
                        tag="ModbusRTUReader"
                    )
                    return False, None
        
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
                    tag="ModbusRTUReader"
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
                tag="ModbusRTUReader"
            )
            return True
        except Exception as e:
            self.logger.log(
                data=f"Failed to save data: {str(e)}",
                log_type=2,
                visibility="TD",
                tag="ModbusRTUReader"
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
            "port": self.port,
            "baudrate": self.baudrate,
            "parity": self.parity,
            "stopbits": self.stopbits,
            "bytesize": self.bytesize,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "is_connected": self.is_connected,
            "connection_attempts": self.connection_attempts,
            "last_read_time": self.last_read_time,
            "register_count": len(self.registers),
            "stats": self.stats.copy()
        }
    
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
                tag="ModbusRTUReader"
            )
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Example usage and backward compatibility
def create_modbus_rtu_reader(
    device_id: str,
    port: str,
    baudrate: int = 9600,
    **kwargs
) -> ModbusRTUReader:
    """
    Factory function to create a Modbus RTU reader.
    
    Args:
        device_id: Unique identifier for the device
        port: Serial port
        baudrate: Baud rate
        **kwargs: Additional arguments
        
    Returns:
        Configured ModbusRTUReader instance
    """
    return ModbusRTUReader(device_id, port, baudrate, **kwargs)


# Backward compatibility function
def read_modbus_rtu_data(
    device_id: str,
    port: str,
    registers: List[ModbusRegister],
    **kwargs
) -> Dict[str, Any]:
    """
    Read data from Modbus RTU device (backward compatibility function).
    
    Args:
        device_id: Device identifier
        port: Serial port
        registers: List of registers to read
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of register values
    """
    reader = ModbusRTUReader(device_id, port, **kwargs)
    reader.add_registers(registers)
    
    with reader:
        return reader.read_registers()