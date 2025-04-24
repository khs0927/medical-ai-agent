// AI consultation routes
// AI-powered consultation endpoints using AI Helper model
app.post('/api/analysis/consultation', async (req, res) => {
  try {
    const { message, userId, healthData } = req.body;
    
    // Log the incoming request
    console.log(`Received consultation request for user ${userId}`);
    
    // Load AI Helper configuration from environment
    const aiHelperEndpoint = process.env.AI_HELPER_ENDPOINT || 'http://localhost:8080';
    const aiHelperApiKey = process.env.AI_HELPER_API_KEY;
    
    // Call AI Helper system
    const response = await fetch(`${aiHelperEndpoint}/api/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${aiHelperApiKey}`
      },
      body: JSON.stringify({
        query: message,
        user_id: userId,
        health_data: healthData
      })
    });
    
    if (!response.ok) {
      throw new Error(`AI Helper returned status ${response.status}`);
    }
    
    const aiResponse = await response.json();
    
    // Return the AI Helper response to the client
    res.json({
      aiResponse: aiResponse.response,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('AI consultation error:', error);
    res.status(500).json({ 
      error: 'Error processing AI consultation',
      message: error.message
    });
  }
});

// Legacy endpoint - redirect to new one
app.post('/api/haim/consultation', (req, res) => {
  console.log('Redirecting from legacy endpoint to new one');
  
  // Forward the request to the new endpoint
  req.url = '/api/analysis/consultation';
  app.handle(req, res);
}); 