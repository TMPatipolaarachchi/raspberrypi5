"""
Real-time Alert Module
Sends alerts when elephants are detected or aggressive behavior is identified
Supports GPIO, email, and webhook notifications
"""

import os
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert data structure"""
    timestamp: datetime
    alert_type: str  # "elephant_detected", "aggressive_behavior", "herd_detected"
    severity: str  # "info", "warning", "critical"
    message: str
    data: Dict
    image_path: Optional[str] = None


class AlertManager:
    """
    Manages alerts for elephant detection system
    Supports multiple notification channels
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize alert manager
        
        Args:
            config: Configuration dictionary with notification settings
        """
        self.config = config or {}
        self.alert_queue = Queue()
        self.alert_history = []
        self.max_history = 100
        
        # Cooldown to prevent alert spam
        self.cooldown_seconds = self.config.get('cooldown_seconds', 30)
        self.last_alert_time = {}
        
        # Notification handlers
        self.handlers: List[Callable] = []
        
        # Setup handlers based on config
        self._setup_handlers()
        
        # Start alert processing thread
        self._running = True
        self._thread = threading.Thread(target=self._process_alerts, daemon=True)
        self._thread.start()
        
        logger.info("Alert manager initialized")
    
    def _setup_handlers(self):
        """Setup notification handlers based on config"""
        
        # GPIO handler (for physical alarm/LED)
        if self.config.get('gpio_enabled', False):
            try:
                self.handlers.append(self._gpio_handler)
                logger.info("GPIO alert handler enabled")
            except Exception as e:
                logger.warning(f"Could not setup GPIO handler: {e}")
        
        # Webhook handler
        if self.config.get('webhook_url'):
            self.handlers.append(self._webhook_handler)
            logger.info("Webhook alert handler enabled")
        
        # Email handler
        if self.config.get('email_enabled', False):
            self.handlers.append(self._email_handler)
            logger.info("Email alert handler enabled")
        
        # Log handler (always enabled)
        self.handlers.append(self._log_handler)
    
    def send_alert(self, alert_type: str, message: str, data: Dict = None,
                   severity: str = "info", image_path: str = None):
        """
        Send an alert
        
        Args:
            alert_type: Type of alert
            message: Alert message
            data: Additional data
            severity: Alert severity (info, warning, critical)
            image_path: Path to captured image
        """
        # Check cooldown
        if not self._check_cooldown(alert_type):
            return
        
        alert = Alert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            data=data or {},
            image_path=image_path
        )
        
        self.alert_queue.put(alert)
        self.last_alert_time[alert_type] = time.time()
    
    def _check_cooldown(self, alert_type: str) -> bool:
        """Check if alert is allowed based on cooldown"""
        last_time = self.last_alert_time.get(alert_type, 0)
        return (time.time() - last_time) >= self.cooldown_seconds
    
    def _process_alerts(self):
        """Process alerts from queue"""
        while self._running:
            try:
                alert = self.alert_queue.get(timeout=1)
                
                # Store in history
                self.alert_history.append(alert)
                if len(self.alert_history) > self.max_history:
                    self.alert_history.pop(0)
                
                # Send to all handlers
                for handler in self.handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")
                
            except:
                continue
    
    def _log_handler(self, alert: Alert):
        """Log alert to file/console"""
        log_level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'critical': logging.CRITICAL
        }.get(alert.severity, logging.INFO)
        
        logger.log(log_level, f"ALERT [{alert.alert_type}]: {alert.message}")
    
    def _gpio_handler(self, alert: Alert):
        """Handle alert via GPIO (LED/buzzer)"""
        try:
            import RPi.GPIO as GPIO
            
            # Configuration
            led_pin = self.config.get('gpio_led_pin', 17)
            buzzer_pin = self.config.get('gpio_buzzer_pin', 27)
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(led_pin, GPIO.OUT)
            GPIO.setup(buzzer_pin, GPIO.OUT)
            
            # Flash pattern based on severity
            if alert.severity == 'critical':
                # Rapid flash for aggressive behavior
                for _ in range(5):
                    GPIO.output(led_pin, GPIO.HIGH)
                    GPIO.output(buzzer_pin, GPIO.HIGH)
                    time.sleep(0.1)
                    GPIO.output(led_pin, GPIO.LOW)
                    GPIO.output(buzzer_pin, GPIO.LOW)
                    time.sleep(0.1)
            elif alert.severity == 'warning':
                # Medium flash for herd
                for _ in range(3):
                    GPIO.output(led_pin, GPIO.HIGH)
                    time.sleep(0.3)
                    GPIO.output(led_pin, GPIO.LOW)
                    time.sleep(0.3)
            else:
                # Single flash for detection
                GPIO.output(led_pin, GPIO.HIGH)
                time.sleep(0.5)
                GPIO.output(led_pin, GPIO.LOW)
            
        except Exception as e:
            logger.error(f"GPIO alert error: {e}")
    
    def _webhook_handler(self, alert: Alert):
        """Send alert to webhook URL"""
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                return
            
            payload = {
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'data': alert.data
            }
            
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"Webhook returned status {response.status}")
            
        except Exception as e:
            logger.error(f"Webhook alert error: {e}")
    
    def _email_handler(self, alert: Alert):
        """Send alert via email"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            smtp_server = self.config.get('smtp_server')
            smtp_port = self.config.get('smtp_port', 587)
            username = self.config.get('smtp_username')
            password = self.config.get('smtp_password')
            recipients = self.config.get('email_recipients', [])
            
            if not all([smtp_server, username, password, recipients]):
                return
            
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[Elephant Alert] {alert.alert_type.upper()}: {alert.severity}"
            
            body = f"""
Elephant Detection Alert

Time: {alert.timestamp}
Type: {alert.alert_type}
Severity: {alert.severity}

Message: {alert.message}

Data: {json.dumps(alert.data, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
        except Exception as e:
            logger.error(f"Email alert error: {e}")
    
    def stop(self):
        """Stop alert manager"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)


class ElephantAlertIntegration:
    """
    Integration class for elephant detection alerts
    Monitors detection results and generates appropriate alerts
    """
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.previous_state = {
            'elephant_detected': False,
            'behavior': 'calm',
            'group': 'none'
        }
    
    def process_result(self, detection_result, behavior_result, image_path: str = None):
        """
        Process detection result and generate alerts as needed
        
        Args:
            detection_result: ElephantDetection result
            behavior_result: BehaviorResult
            image_path: Path to saved frame image
        """
        current_state = {
            'elephant_detected': detection_result.elephant_detected,
            'behavior': behavior_result.behavior if behavior_result else 'unknown',
            'group': detection_result.group_classification
        }
        
        # Check for elephant detection (new detection)
        if current_state['elephant_detected'] and not self.previous_state['elephant_detected']:
            self.alert_manager.send_alert(
                alert_type='elephant_detected',
                message=f"Elephant detected! Adults: {detection_result.adult_count}, Calves: {detection_result.calf_count}",
                data={
                    'adult_count': detection_result.adult_count,
                    'calf_count': detection_result.calf_count,
                    'group': detection_result.group_classification
                },
                severity='info',
                image_path=image_path
            )
        
        # Check for aggressive behavior
        if (current_state['behavior'] == 'aggressive' and 
            self.previous_state['behavior'] != 'aggressive'):
            self.alert_manager.send_alert(
                alert_type='aggressive_behavior',
                message="Warning: Aggressive elephant behavior detected!",
                data={
                    'confidence': behavior_result.confidence if behavior_result else 0,
                    'pose_behavior': behavior_result.pose_behavior if behavior_result else 'unknown',
                    'sound_behavior': behavior_result.sound_behavior if behavior_result else 'unknown'
                },
                severity='critical',
                image_path=image_path
            )
        
        # Check for herd detection
        if (current_state['group'] == 'herd' and 
            self.previous_state['group'] != 'herd'):
            self.alert_manager.send_alert(
                alert_type='herd_detected',
                message=f"Elephant herd detected! Total: {detection_result.total_count}",
                data={
                    'adult_count': detection_result.adult_count,
                    'calf_count': detection_result.calf_count,
                    'total_count': detection_result.total_count
                },
                severity='warning',
                image_path=image_path
            )
        
        self.previous_state = current_state


# Default alert configuration
DEFAULT_ALERT_CONFIG = {
    'cooldown_seconds': 30,
    'gpio_enabled': False,
    'gpio_led_pin': 17,
    'gpio_buzzer_pin': 27,
    'webhook_url': None,  # Set your webhook URL here
    'email_enabled': False,
    'smtp_server': None,
    'smtp_port': 587,
    'smtp_username': None,
    'smtp_password': None,
    'email_recipients': []
}


if __name__ == "__main__":
    # Test alert system
    logging.basicConfig(level=logging.INFO)
    
    config = DEFAULT_ALERT_CONFIG.copy()
    manager = AlertManager(config)
    
    # Send test alerts
    manager.send_alert(
        alert_type='elephant_detected',
        message='Test: Elephant detected!',
        data={'adult_count': 2, 'calf_count': 1},
        severity='info'
    )
    
    time.sleep(1)
    
    manager.send_alert(
        alert_type='aggressive_behavior',
        message='Test: Aggressive behavior!',
        data={'confidence': 0.95},
        severity='critical'
    )
    
    time.sleep(2)
    manager.stop()
    
    print("Alert system test complete")
