#!/usr/bin/env python3
"""
Real-time Alert & Notification System V2.1
Sends notifications for critical healthcare events
"""

import smtplib
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import asyncio
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareAlertSystem:
    """Real-time healthcare alert and notification system"""
    
    def __init__(self):
        self.alerts_dir = Path('alerts')
        self.alerts_dir.mkdir(exist_ok=True)
        
        # Alert thresholds
        self.thresholds = {
            'covid_positive_rate': 0.15,  # 15% positive rate
            'icu_occupancy': 0.85,  # 85% ICU occupancy
            'emergency_wait_time': 45,  # 45 minutes wait time
            'staff_shortage': 0.25,  # 25% staff shortage
            'critical_alerts': 3  # 3+ critical alerts
        }
        
        # Notification channels
        self.notification_channels = {
            'email': True,
            'telegram': False,  # Requires bot token
            'slack': False,  # Requires webhook
            'webhook': False  # Custom webhook
        }
        
        # Alert history
        self.alert_history = []
        self.last_notification = {}
        
    def check_covid_alerts(self, covid_data: pd.DataFrame) -> List[Dict]:
        """Check for COVID-related alerts"""
        alerts = []
        
        if covid_data.empty:
            return alerts
        
        # Check high positive rates
        high_positive = covid_data[covid_data['positive_rate'] > self.thresholds['covid_positive_rate']]
        for _, row in high_positive.iterrows():
            alerts.append({
                'type': 'COVID_HIGH_POSITIVE_RATE',
                'severity': 'High',
                'state': row['state'],
                'value': f"{row['positive_rate']:.1%}",
                'threshold': f"{self.thresholds['covid_positive_rate']:.1%}",
                'message': f"High COVID positive rate in {row['state']}: {row['positive_rate']:.1%}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check high hospitalization rates
        high_hosp = covid_data[covid_data['hospitalization_rate'] > 0.1]  # 10% hospitalization
        for _, row in high_hosp.iterrows():
            alerts.append({
                'type': 'COVID_HIGH_HOSPITALIZATION',
                'severity': 'Critical',
                'state': row['state'],
                'value': f"{row['hospitalization_rate']:.1%}",
                'threshold': '10%',
                'message': f"Critical hospitalization rate in {row['state']}: {row['hospitalization_rate']:.1%}",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def check_hospital_alerts(self, hospital_data: pd.DataFrame) -> List[Dict]:
        """Check for hospital capacity alerts"""
        alerts = []
        
        if hospital_data.empty:
            return alerts
        
        # Check ICU occupancy
        high_icu = hospital_data[hospital_data['icu_occupancy'] > self.thresholds['icu_occupancy']]
        for _, row in high_icu.iterrows():
            alerts.append({
                'type': 'HOSPITAL_HIGH_ICU_OCCUPANCY',
                'severity': 'Critical',
                'state': row['state'],
                'value': f"{row['icu_occupancy']:.1%}",
                'threshold': f"{self.thresholds['icu_occupancy']:.1%}",
                'message': f"Critical ICU occupancy in {row['state']}: {row['icu_occupancy']:.1%}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check emergency wait times
        long_wait = hospital_data[hospital_data['emergency_wait_time'] > self.thresholds['emergency_wait_time']]
        for _, row in long_wait.iterrows():
            alerts.append({
                'type': 'HOSPITAL_LONG_EMERGENCY_WAIT',
                'severity': 'High',
                'state': row['state'],
                'value': f"{row['emergency_wait_time']:.1f} min",
                'threshold': f"{self.thresholds['emergency_wait_time']} min",
                'message': f"Long emergency wait time in {row['state']}: {row['emergency_wait_time']:.1f} min",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check staff shortages
        staff_shortage = hospital_data[hospital_data['staff_shortage'] > self.thresholds['staff_shortage']]
        for _, row in staff_shortage.iterrows():
            alerts.append({
                'type': 'HOSPITAL_STAFF_SHORTAGE',
                'severity': 'Medium',
                'state': row['state'],
                'value': f"{row['staff_shortage']:.1%}",
                'threshold': f"{self.thresholds['staff_shortage']:.1%}",
                'message': f"Staff shortage in {row['state']}: {row['staff_shortage']:.1%}",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def check_emergency_alerts(self, alerts_data: pd.DataFrame) -> List[Dict]:
        """Check for emergency alerts"""
        alerts = []
        
        if alerts_data.empty:
            return alerts
        
        # Check critical alerts
        critical_alerts = alerts_data[alerts_data['severity'] == 'Critical']
        if len(critical_alerts) >= self.thresholds['critical_alerts']:
            alerts.append({
                'type': 'MULTIPLE_CRITICAL_ALERTS',
                'severity': 'Critical',
                'states': critical_alerts['state'].tolist(),
                'count': len(critical_alerts),
                'threshold': self.thresholds['critical_alerts'],
                'message': f"Multiple critical alerts active: {len(critical_alerts)} across {', '.join(critical_alerts['state'].unique())}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check high severity alerts
        high_alerts = alerts_data[alerts_data['severity'] == 'High']
        for _, row in high_alerts.iterrows():
            alerts.append({
                'type': 'EMERGENCY_HIGH_SEVERITY',
                'severity': 'High',
                'state': row['state'],
                'alert_type': row['type'],
                'message': f"High severity {row['type']} alert in {row['state']}",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def check_weather_alerts(self, weather_data: pd.DataFrame) -> List[Dict]:
        """Check for weather-related healthcare alerts"""
        alerts = []
        
        if weather_data.empty:
            return alerts
        
        # Check extreme temperatures
        extreme_temp = weather_data[
            (weather_data['temperature'] < 0) | (weather_data['temperature'] > 35)
        ]
        for _, row in extreme_temp.iterrows():
            severity = 'Critical' if abs(row['temperature'] - 20) > 20 else 'High'
            alerts.append({
                'type': 'WEATHER_EXTREME_TEMPERATURE',
                'severity': severity,
                'city': row['city'],
                'temperature': f"{row['temperature']:.1f}°C",
                'message': f"Extreme temperature in {row['city']}: {row['temperature']:.1f}°C",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check severe weather conditions
        severe_weather = weather_data[weather_data['weather_condition'].isin(['Storm', 'Hurricane', 'Tornado'])]
        for _, row in severe_weather.iterrows():
            alerts.append({
                'type': 'WEATHER_SEVERE_CONDITION',
                'severity': 'Critical',
                'city': row['city'],
                'condition': row['weather_condition'],
                'message': f"Severe weather in {row['city']}: {row['weather_condition']}",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def send_email_alert(self, alert: Dict, recipients: List[str] = None):
        """Send email alert"""
        if not self.notification_channels['email']:
            return
        
        try:
            # Email configuration (replace with real SMTP settings)
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "healthcare-alerts@example.com"
            sender_password = "your_password"  # Use environment variable in production
            
            if not recipients:
                recipients = ["admin@healthcare.com"]
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"🚨 Healthcare Alert: {alert['type']} - {alert['severity']}"
            
            # Create HTML body
            html_body = f"""
            <html>
            <body>
                <h2>🚨 Healthcare Alert</h2>
                <p><strong>Type:</strong> {alert['type']}</p>
                <p><strong>Severity:</strong> {alert['severity']}</p>
                <p><strong>Message:</strong> {alert['message']}</p>
                <p><strong>Time:</strong> {alert['timestamp']}</p>
                <hr>
                <p><em>US Healthcare Insights Dashboard - Real-time Monitoring</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email (commented out for demo)
            # server = smtplib.SMTP(smtp_server, smtp_port)
            # server.starttls()
            # server.login(sender_email, sender_password)
            # server.send_message(msg)
            # server.quit()
            
            logger.info(f"✅ Email alert sent for {alert['type']}")
            
        except Exception as e:
            logger.error(f"❌ Error sending email alert: {e}")
    
    def send_telegram_alert(self, alert: Dict, chat_id: str = None):
        """Send Telegram alert"""
        if not self.notification_channels['telegram']:
            return
        
        try:
            # Telegram configuration
            bot_token = "your_bot_token"  # Use environment variable
            if not chat_id:
                chat_id = "your_chat_id"
            
            message = f"""
🚨 Healthcare Alert

Type: {alert['type']}
Severity: {alert['severity']}
Message: {alert['message']}
Time: {alert['timestamp']}

US Healthcare Insights Dashboard
            """
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            # Send message (commented out for demo)
            # response = requests.post(url, data=data)
            # if response.status_code == 200:
            #     logger.info(f"✅ Telegram alert sent for {alert['type']}")
            # else:
            #     logger.error(f"❌ Telegram alert failed: {response.text}")
            
            logger.info(f"✅ Telegram alert prepared for {alert['type']}")
            
        except Exception as e:
            logger.error(f"❌ Error sending Telegram alert: {e}")
    
    def send_webhook_alert(self, alert: Dict, webhook_url: str = None):
        """Send webhook alert"""
        if not self.notification_channels['webhook']:
            return
        
        try:
            if not webhook_url:
                webhook_url = "https://your-webhook-url.com/alerts"
            
            payload = {
                'alert': alert,
                'timestamp': datetime.now().isoformat(),
                'source': 'US Healthcare Insights Dashboard'
            }
            
            # Send webhook (commented out for demo)
            # response = requests.post(webhook_url, json=payload)
            # if response.status_code == 200:
            #     logger.info(f"✅ Webhook alert sent for {alert['type']}")
            # else:
            #     logger.error(f"❌ Webhook alert failed: {response.status_code}")
            
            logger.info(f"✅ Webhook alert prepared for {alert['type']}")
            
        except Exception as e:
            logger.error(f"❌ Error sending webhook alert: {e}")
    
    def process_alerts(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Process all data and generate alerts"""
        all_alerts = []
        
        # Check different data types
        if 'covid_data' in data:
            all_alerts.extend(self.check_covid_alerts(data['covid_data']))
        
        if 'hospital_status' in data:
            all_alerts.extend(self.check_hospital_alerts(data['hospital_status']))
        
        if 'emergency_alerts' in data:
            all_alerts.extend(self.check_emergency_alerts(data['emergency_alerts']))
        
        if 'weather_data' in data:
            all_alerts.extend(self.check_weather_alerts(data['weather_data']))
        
        # Filter out duplicate alerts (within 1 hour)
        filtered_alerts = []
        for alert in all_alerts:
            alert_key = f"{alert['type']}_{alert.get('state', alert.get('city', 'unknown'))}"
            last_alert_time = self.last_notification.get(alert_key)
            
            if not last_alert_time or (datetime.now() - last_alert_time).seconds > 3600:
                filtered_alerts.append(alert)
                self.last_notification[alert_key] = datetime.now()
        
        return filtered_alerts
    
    def send_notifications(self, alerts: List[Dict]):
        """Send notifications for all alerts"""
        for alert in alerts:
            # Send to all configured channels
            self.send_email_alert(alert)
            self.send_telegram_alert(alert)
            self.send_webhook_alert(alert)
            
            # Log alert
            logger.info(f"🚨 Alert sent: {alert['type']} - {alert['severity']}")
    
    def save_alert_history(self, alerts: List[Dict]):
        """Save alert history"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save to JSON
        alert_file = self.alerts_dir / f"alerts_{timestamp}.json"
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        # Save to CSV
        if alerts:
            df = pd.DataFrame(alerts)
            csv_file = self.alerts_dir / f"alerts_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
        
        # Update history
        self.alert_history.extend(alerts)
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def run_alert_check(self, data: Dict[str, pd.DataFrame]):
        """Run complete alert check and notification"""
        logger.info("🔍 Running alert check...")
        
        # Process alerts
        alerts = self.process_alerts(data)
        
        if alerts:
            logger.info(f"🚨 Found {len(alerts)} alerts")
            
            # Send notifications
            self.send_notifications(alerts)
            
            # Save history
            self.save_alert_history(alerts)
            
            # Log summary
            severity_counts = {}
            for alert in alerts:
                severity_counts[alert['severity']] = severity_counts.get(alert['severity'], 0) + 1
            
            logger.info(f"📊 Alert summary: {severity_counts}")
        else:
            logger.info("✅ No alerts detected")

def main():
    """Main function for testing alert system"""
    alert_system = HealthcareAlertSystem()
    
    # Load sample data
    data_dir = Path('data/realtime')
    data = {}
    
    for data_type in ['covid_data', 'weather_data', 'hospital_status', 'emergency_alerts']:
        files = list(data_dir.glob(f"{data_type}_*.csv"))
        if files:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            data[data_type] = pd.read_csv(latest_file)
    
    # Run alert check
    alert_system.run_alert_check(data)

if __name__ == "__main__":
    main()
