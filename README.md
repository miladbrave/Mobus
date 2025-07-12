# Modbus Communication Library

A standalone Python library for Modbus RTU and TCP communication with IoT devices. This library provides clean, object-oriented interfaces for both serial (RTU) and network (TCP) Modbus communication without external framework dependencies.

## Overview

This project provides two main classes:
- **ModbusRTUReader**: For serial communication over RS-485/RS-232
- **ModbusTCPReader**: For network communication over TCP/IP

Both classes include connection management, health monitoring, data transformation, and comprehensive error handling.

## Features

- **Standalone Implementation**: No external framework dependencies
- **Dual Protocol Support**: RTU (serial) and TCP (network) communication
- **Connection Management**: Automatic connection handling with retry logic
- **Health Monitoring**: Built-in health checks and connection monitoring
- **Data Transformation**: Support for multiple data types with automatic conversion
- **Statistics Tracking**: Detailed performance and usage statistics
- **Context Manager Support**: Safe resource management with `with` statements
- **Thread Safety**: Thread-safe operations for TCP connections
- **Connection Pooling**: Efficient connection reuse for TCP communication

## Installation

### Prerequisites

```bash
# For Modbus communication
pip install pymodbus

# For serial communication (Modbus RTU)
pip install pyserial
```

### Usage

Simply copy the desired class files into your project and import them:

```python
from modbus_rtu import ModbusRTUReader, ModbusRegister, ModbusDataType
from modbus_tcp import ModbusTCPReader, ModbusRegister, ModbusDataType
```

## Modbus RTU Reader

The Modbus RTU Reader provides serial communication with Modbus RTU devices over RS-485/RS-232 connections.

### Features

- Serial communication with configurable parameters
- Support for multiple data types (INT16, INT32, FLOAT32, FLOAT64, STRING, BOOLEAN)
- Automatic connection management and retry logic
- Data transformation and scaling
- Health monitoring and statistics

### Quick Start

```python
from modbus_rtu import ModbusRTUReader, ModbusRegister, ModbusDataType

# Create Modbus RTU reader
reader = ModbusRTUReader(
    device_id="device_001",
    port="/dev/ttyUSB0",
    baudrate=9600,
    parity="N",
    timeout=1.0
)

# Add registers to read
voltage_register = ModbusRegister(
    address=1000,
    data_type=ModbusDataType.FLOAT32,
    name="voltage",
    description="Line voltage",
    unit="V",
    scale_factor=1.0
)
reader.add_register(voltage_register)

# Read data
with reader:
    data = reader.read_registers()
    print(data)
```

### Serial Configuration

- **Port**: Serial port (e.g., `/dev/ttyUSB0`, `COM1`)
- **Baudrate**: Communication speed (default: 9600)
- **Parity**: Parity bit (`N`, `E`, `O`)
- **Stopbits**: Number of stop bits (default: 1)
- **Bytesize**: Number of data bits (default: 8)

### Supported Data Types

- **INT16**: 16-bit integer (1 register)
- **INT32**: 32-bit integer (2 registers)
- **FLOAT32**: 32-bit float (2 registers)
- **FLOAT64**: 64-bit float (4 registers)
- **STRING**: ASCII string (variable registers)
- **BOOLEAN**: Boolean value (1 register)

## Modbus TCP Reader

The Modbus TCP Reader provides network communication with Modbus TCP devices over Ethernet.

### Features

- TCP/IP communication with connection pooling
- Support for multiple data types
- Automatic connection management and health monitoring
- Thread-safe operations
- Performance optimization with connection reuse

### Quick Start

```python
from modbus_tcp import ModbusTCPReader, ModbusRegister, ModbusDataType

# Create Modbus TCP reader
reader = ModbusTCPReader(
    device_id="device_001",
    host="192.168.1.100",
    port=502,
    timeout=1.0,
    max_connections=5
)

# Add registers to read
power_register = ModbusRegister(
    address=2000,
    data_type=ModbusDataType.FLOAT32,
    name="power",
    description="Active power",
    unit="W",
    scale_factor=1.0
)
reader.add_register(power_register)

# Read data
data = reader.read_registers()
print(data)
```

### Connection Pooling

The Modbus TCP Reader uses connection pooling to improve performance:
- **Max Connections**: Configurable pool size (default: 5)
- **Connection Reuse**: Automatic connection management
- **Health Checks**: Regular connection health monitoring
- **Automatic Recovery**: Reconnection on connection failures

## Configuration

### Logging

All classes use a simple logging system that can be customized:

```python
from modbus_rtu import SimpleLogger

# Create custom logger
logger = SimpleLogger(log_level=1)  # 0=info, 1=warning, 2=error

# Use with any class
reader = ModbusRTUReader(device_id="test", logger=logger)
```

### Health Monitoring

Health monitoring is automatically enabled for all classes:

```python
# Check health status
status = reader.get_status()
print(f"Health: {status['health_status']}")
print(f"Last check: {status['last_health_check']}")
```

### Statistics

All classes provide detailed statistics:

```python
# Get statistics
stats = reader.get_status()['stats']
print(f"Total reads: {stats['total_reads']}")
print(f"Successful reads: {stats['successful_reads']}")
print(f"Failed reads: {stats['failed_reads']}")
```

## Examples

### Complete Modbus RTU Example

```python
from modbus_rtu import ModbusRTUReader, ModbusRegister, ModbusDataType

# Create reader
reader = ModbusRTUReader(
    device_id="sensor_001",
    port="/dev/ttyUSB0",
    baudrate=9600,
    timeout=1.0,
    retry_count=3
)

# Add sensor registers
registers = [
    ModbusRegister(1000, ModbusDataType.FLOAT32, "temperature", "Temperature", "°C"),
    ModbusRegister(1002, ModbusDataType.FLOAT32, "humidity", "Humidity", "%"),
    ModbusRegister(1004, ModbusDataType.FLOAT32, "pressure", "Pressure", "hPa")
]

for register in registers:
    reader.add_register(register)

# Read data with context manager
with reader:
    data = reader.read_registers()
    print("Sensor Data:", data)
    
    # Check connection health
    if reader.check_health():
        print("Connection is healthy")
    else:
        print("Connection issues detected")
```

### Complete Modbus TCP Example

```python
from modbus_tcp import ModbusTCPReader, ModbusRegister, ModbusDataType

# Create reader
reader = ModbusTCPReader(
    device_id="plc_001",
    host="192.168.1.100",
    port=502,
    timeout=1.0,
    max_connections=5,
    retry_count=3
)

# Add PLC registers
registers = [
    ModbusRegister(2000, ModbusDataType.FLOAT32, "motor_speed", "Motor Speed", "RPM"),
    ModbusRegister(2002, ModbusDataType.FLOAT32, "motor_current", "Motor Current", "A"),
    ModbusRegister(2004, ModbusDataType.FLOAT32, "motor_voltage", "Motor Voltage", "V"),
    ModbusRegister(2006, ModbusDataType.BOOLEAN, "motor_status", "Motor Status", "")
]

for register in registers:
    reader.add_register(register)

# Read data
data = reader.read_registers()
print("PLC Data:", data)

# Get connection pool status
status = reader.get_status()
pool_stats = status['pool_stats']
print(f"Active connections: {pool_stats['active_connections']}")
print(f"Pool size: {pool_stats['pool_size']}")
```

### Industrial Automation Example

```python
from modbus_tcp import ModbusTCPReader, ModbusRegister, ModbusDataType
import time

# Create reader for industrial equipment
reader = ModbusTCPReader(
    device_id="production_line_01",
    host="10.0.1.50",
    port=502,
    timeout=2.0,
    max_connections=10
)

# Add production line registers
registers = [
    ModbusRegister(3000, ModbusDataType.INT32, "production_count", "Production Count", "units"),
    ModbusRegister(3002, ModbusDataType.FLOAT32, "temperature", "Process Temperature", "°C"),
    ModbusRegister(3004, ModbusDataType.FLOAT32, "pressure", "Process Pressure", "bar"),
    ModbusRegister(3006, ModbusDataType.BOOLEAN, "emergency_stop", "Emergency Stop", ""),
    ModbusRegister(3007, ModbusDataType.BOOLEAN, "system_ready", "System Ready", "")
]

for register in registers:
    reader.add_register(register)

# Continuous monitoring
with reader:
    while True:
        try:
            data = reader.read_registers()
            
            # Check for emergency stop
            if data.get('emergency_stop'):
                print("EMERGENCY STOP ACTIVATED!")
                break
            
            # Check system status
            if data.get('system_ready'):
                print(f"Production Count: {data.get('production_count', 0)}")
                print(f"Temperature: {data.get('temperature', 0):.1f}°C")
                print(f"Pressure: {data.get('pressure', 0):.2f} bar")
            else:
                print("System not ready")
            
            time.sleep(5)  # Read every 5 seconds
            
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
            break
```

## Error Handling

All classes include comprehensive error handling:

- **Connection Errors**: Automatic retry with exponential backoff
- **Data Validation**: Input validation and error reporting
- **Resource Management**: Safe cleanup with context managers
- **Logging**: Detailed error logging with different levels

## Performance Considerations

### Modbus RTU
- Use appropriate baudrate for your network (9600, 19200, 38400, 57600, 115200)
- Configure timeout based on device response time
- Consider retry settings for unreliable connections
- Use proper cable shielding for RS-485 networks

### Modbus TCP
- Adjust connection pool size based on load (5-20 connections typically)
- Monitor connection health regularly
- Use appropriate timeout values (1-5 seconds)
- Consider network latency in timeout calculations

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check network connectivity and device addresses
   - Verify serial port permissions (Linux: `sudo usermod -a -G dialout $USER`)
   - Ensure proper cable connections for RS-485

2. **Data Validation Errors**
   - Check register addresses and data types
   - Verify byte order settings (big-endian vs little-endian)
   - Validate scale factors for data transformation

3. **Performance Issues**
   - Adjust timeout and retry settings
   - Monitor connection pool usage
   - Check for resource leaks

### Debug Mode

Enable debug logging by setting log level to 0:

```python
logger = SimpleLogger(log_level=0)
```

### Serial Port Issues (RTU)

```bash
# List available serial ports (Linux)
ls /dev/tty*

# Check serial port permissions
ls -l /dev/ttyUSB0

# Add user to dialout group (Linux)
sudo usermod -a -G dialout $USER
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the examples
3. Create an issue with detailed information

## Version History

- **v1.0.0**: Initial release with Modbus RTU and TCP support
- Standalone implementation without external framework dependencies
- Comprehensive error handling and health monitoring
- Thread-safe operations and context manager support
- Connection pooling for TCP communication 
