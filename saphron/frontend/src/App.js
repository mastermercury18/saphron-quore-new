import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import axios from 'axios';
import { 
  Brain, 
  CheckCircle, 
  XCircle, 
  TrendingUp, 
  BookOpen, 
  Target,
  Loader2,
  Sparkles
} from 'lucide-react';

// Import shadcn/ui components
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Progress } from './components/ui/progress';
import { Badge } from './components/ui/badge';
import { Alert, AlertDescription } from './components/ui/alert';

function App() {
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [knowledge, setKnowledge] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    fetchQuestion();
  }, []);

  const fetchQuestion = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:5001/api/question');
      setCurrentQuestion(response.data);
      setSelectedAnswer(null);
      setFeedback(null);
      setKnowledge(response.data.knowledge || []);
    } catch (error) {
      console.error('Error fetching question:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (selectedAnswer === null) return;
    
    try {
      setSubmitting(true);
      const response = await axios.post('http://localhost:5001/api/submit', {
        answer: selectedAnswer
      });
      
      setFeedback(response.data);
      setKnowledge(response.data.knowledge || []);
    } catch (error) {
      console.error('Error submitting answer:', error);
    } finally {
      setSubmitting(false);
    }
  };

  const handleNextQuestion = () => {
    fetchQuestion();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="relative">
            <Loader2 className="h-12 w-12 animate-spin text-blue-600 mx-auto" />
            <Sparkles className="h-6 w-6 text-purple-500 absolute -top-2 -right-2 animate-pulse" />
          </div>
          <h2 className="text-xl font-semibold text-gray-700">Loading your learning session...</h2>
          <p className="text-gray-500">Preparing quantum-inspired questions</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="relative">
              <Brain className="h-12 w-12 text-blue-600" />
              <Sparkles className="h-6 w-6 text-purple-500 absolute -top-2 -right-2 animate-pulse" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Adaptive Learning System
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Powered by Quantum-Inspired Reinforcement Learning
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Question Area */}
          <div className="lg:col-span-2">
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <BookOpen className="h-5 w-5 text-blue-600" />
                    <CardTitle className="text-xl">
                      Question #{currentQuestion?.qid + 1}
                    </CardTitle>
                  </div>
                  <Badge variant="secondary" className="text-sm">
                    Topic {currentQuestion?.topic}
                  </Badge>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-6">
                {currentQuestion && (
                  <>
                    <div className="prose prose-lg max-w-none">
                      <p className="text-gray-700 leading-relaxed text-lg">
                        {currentQuestion.question}
                      </p>
                    </div>

                    {!feedback && (
                      <div className="space-y-4">
                        <div className="grid gap-3">
                          {currentQuestion.options.map((option, index) => (
                            <button
                              key={index}
                              onClick={() => setSelectedAnswer(index)}
                              className={`p-4 text-left rounded-lg border-2 transition-all duration-200 hover:shadow-md ${
                                selectedAnswer === index
                                  ? 'border-blue-500 bg-blue-50 shadow-md'
                                  : 'border-gray-200 bg-white hover:border-gray-300'
                              }`}
                            >
                              <div className="flex items-center space-x-3">
                                <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${
                                  selectedAnswer === index
                                    ? 'border-blue-500 bg-blue-500'
                                    : 'border-gray-300'
                                }`}>
                                  {selectedAnswer === index && (
                                    <div className="w-2 h-2 bg-white rounded-full" />
                                  )}
                                </div>
                                <span className="text-gray-700 font-medium">{option}</span>
                              </div>
                            </button>
                          ))}
                        </div>

                        <Button
                          onClick={handleSubmit}
                          disabled={selectedAnswer === null || submitting}
                          className="w-full h-12 text-lg font-semibold bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                        >
                          {submitting ? (
                            <>
                              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                              Processing...
                            </>
                          ) : (
                            'Submit Answer'
                          )}
                        </Button>
                      </div>
                    )}

                    {feedback && (
                      <div className="space-y-4">
                        <Alert className={`border-2 ${
                          feedback.correct 
                            ? 'border-green-200 bg-green-50' 
                            : 'border-red-200 bg-red-50'
                        }`}>
                          <div className="flex items-center space-x-2">
                            {feedback.correct ? (
                              <CheckCircle className="h-5 w-5 text-green-600" />
                            ) : (
                              <XCircle className="h-5 w-5 text-red-600" />
                            )}
                            <AlertDescription className="text-lg font-medium">
                              {feedback.feedback}
                            </AlertDescription>
                          </div>
                        </Alert>

                        <Button
                          onClick={handleNextQuestion}
                          className="w-full h-12 text-lg font-semibold bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
                        >
                          Next Question
                        </Button>
                      </div>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Knowledge Progress Sidebar */}
          <div className="space-y-6">
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <TrendingUp className="h-5 w-5 text-blue-600" />
                  <CardTitle>Knowledge Progress</CardTitle>
                </div>
                <CardDescription>
                  Your mastery level across different topics
                </CardDescription>
              </CardHeader>
              
              <CardContent className="space-y-4">
                {knowledge.length > 0 && (
                  <div className="space-y-4">
                    {knowledge.map((score, index) => (
                      <div key={index} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <Target className="h-4 w-4 text-gray-500" />
                            <span className="text-sm font-medium text-gray-700">
                              Topic {index}
                            </span>
                          </div>
                          <span className="text-sm font-bold text-gray-900">
                            {(score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <Progress 
                          value={score * 100} 
                          className="h-2"
                        />
                      </div>
                    ))}
                  </div>
                )}

                {/* Knowledge Chart */}
                {knowledge.length > 0 && (
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                      Progress Overview
                    </h3>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart 
                          data={knowledge.map((score, index) => ({ 
                            topic: `T${index}`, 
                            score: score * 100 
                          }))}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                          <XAxis 
                            dataKey="topic" 
                            tick={{ fontSize: 12 }}
                            stroke="#6b7280"
                          />
                          <YAxis 
                            tick={{ fontSize: 12 }}
                            stroke="#6b7280"
                          />
                          <Tooltip 
                            formatter={(value) => [`${value.toFixed(1)}%`, 'Mastery']}
                            contentStyle={{
                              backgroundColor: 'white',
                              border: '1px solid #e5e7eb',
                              borderRadius: '8px',
                              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                            }}
                          />
                          <Bar 
                            dataKey="score" 
                            fill="url(#gradient)"
                            radius={[4, 4, 0, 0]}
                          />
                          <defs>
                            <linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#3b82f6" />
                              <stop offset="100%" stopColor="#8b5cf6" />
                            </linearGradient>
                          </defs>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
