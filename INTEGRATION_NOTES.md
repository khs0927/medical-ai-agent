# NotToday App and AI Helper Medical Diagnosis System Integration Analysis

## Integration Structure Analysis

NotToday App and AI Helper Medical Diagnosis System (ADK) are maintained as two separate repositories and integrated via API. This document explains the integration approach and methods for connecting them on GitHub.

## 1. Integration Architecture

### 1.1 System Components

```
[NotToday App (Client)] ←→ [NotToday Server] ←→ [AI Helper Medical Diagnosis System (ADK)]
```

- **NotToday App (Client)**: Provides user interface and ECG data collection functionality
- **NotToday Server**: Acts as an intermediary between the client and AI Helper system
- **AI Helper Medical Diagnosis System (ADK)**: Provides medical data analysis and diagnosis functionality

### 1.2 Communication Flow

1. ECG data collection and user inquiries in the NotToday app
2. Data transmission to NotToday server (API endpoint: `/analysis/consultation`)
3. NotToday server calls the AI Helper system
4. AI Helper system generates analysis results
5. Results are returned to the client via the NotToday server

## 2. API Specification

### 2.1 NotToday Client-Server Communication

**Request (Client → Server)**:
```javascript
// POST /api/analysis/consultation
{
  "message": "My heart is pounding today. Is this normal?",
  "userId": "user123",
  "healthData": {
    "heartRate": 72,
    "oxygenLevel": 98,
    "bloodPressure": {
      "systolic": 120,
      "diastolic": 80
    }
  }
}
```

**Response (Server → Client)**:
```javascript
{
  "aiResponse": "A heart rate of 72bpm is within the normal range (60-100bpm)...",
  "timestamp": "2023-08-01T12:34:56Z"
}
```

### 2.2 NotToday Server-AI Helper Communication

**Request (Server → AI Helper)**:
```python
# ADK system call
response = MedicalCoordinatorAgent.process(
    query=message,
    context={
        "user_id": userId,
        "health_data": healthData
    }
)
```

## 3. Integration on GitHub

The two systems are managed as separate GitHub repositories but can be integrated using the following methods:

### 3.1 Repository Structure

```
github.com/user/NotToday        # NotToday App repository
github.com/user/medical-agent   # AI Helper Medical Diagnosis System repository
```

### 3.2 Integration Options

You can integrate using one of the following three methods:

#### Option 1: Git Submodules (Recommended)

NotToday repository references the AI Helper system as a Git Submodule:

```bash
# Inside the NotToday repository
git submodule add https://github.com/user/medical-agent.git ai-helper
git commit -m "Add medical agent as submodule"
```

This approach allows independent management of code for both systems while fixing to a specific version.

#### Option 2: Docker Container Integration

Deploy the AI Helper system as a Docker container and call it via API from the NotToday server:

```yaml
# docker-compose.yml
services:
  nottoday-server:
    image: nottoday-server:latest
    ports:
      - "8000:8000"
  
  ai-helper:
    image: medical-agent:latest
    ports:
      - "8080:8080"
```

#### Option 3: Independent Deployment and API Integration

Deploy the two systems completely separately and communicate via API:

1. Deploy the AI Helper system on a separate server
2. Specify the AI Helper API endpoint in NotToday's environment configuration
3. NotToday server calls the remote AI Helper system via HTTP/HTTPS

## 4. Environment Configuration

### 4.1 NotToday Environment Variables

```
# NotToday/.env
AI_HELPER_ENDPOINT=http://localhost:8080
AI_HELPER_API_KEY=your_api_key_here
```

### 4.2 AI Helper Environment Variables

```
# medical-agent/.env
HUGGINGFACE_TOKEN=your_hf_token_here
SERVING_PORT=8080
```

## 5. Deployment Scenario Analysis

### 5.1 Single Server Deployment

Suitable for small-scale deployment:
- Deploy NotToday server and AI Helper system on the same server
- Manage containers with Docker Compose
- Minimize latency with local network communication

### 5.2 Distributed Server Deployment

Suitable for large-scale deployment:
- NotToday server: Web frontend and API processing
- AI Helper system: Medical analysis processing on a separate high-performance server
- Possible to manage multiple AI Helper instances with a load balancer

## 6. Conclusion

The NotToday app and AI Helper Medical Diagnosis System can be effectively integrated despite being maintained as separate repositories on GitHub. The Git Submodule approach is most recommended for code management and version compatibility. The two systems can effectively interact through API-based communication and be easily configured through environment variables.

Since the `/api/analysis/consultation` endpoint of NotToday has already been changed to communicate with the AI Helper system, additional code modifications during the integration process can be minimized.

## 7. Next Steps

1. Create GitHub repository for AI Helper Medical Diagnosis System
2. Add AI Helper system as a submodule to NotToday repository
3. Configure environment variables and run integration tests
4. Adjust API endpoints if necessary 