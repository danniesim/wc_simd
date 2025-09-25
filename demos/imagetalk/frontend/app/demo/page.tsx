'use client';

import Alert from '@mui/material/Alert';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type ConversationEntry = {
    role: 'user' | 'assistant';
    text: string;
};

const API_BASE = process.env.NEXT_PUBLIC_IMAGETALK_API_BASE_URL ?? 'http://localhost:8000';

const getAudioContext = (): AudioContext => {
    if (typeof window === 'undefined') {
        throw new Error('AudioContext is not available during SSR.');
    }
    const AudioContextImpl = window.AudioContext ?? (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextImpl) {
        throw new Error('Web Audio API is not supported in this browser.');
    }
    return new AudioContextImpl();
};

const encodeWav = (samples: Float32Array, sampleRate: number): ArrayBuffer => {
    const bytesPerSample = 2;
    const blockAlign = bytesPerSample * 1; // mono audio
    const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
    const view = new DataView(buffer);
    let offset = 0;

    const writeString = (text: string) => {
        for (let i = 0; i < text.length; i += 1) {
            view.setUint8(offset + i, text.charCodeAt(i));
        }
        offset += text.length;
    };

    writeString('RIFF');
    view.setUint32(offset, 36 + samples.length * bytesPerSample, true);
    offset += 4;
    writeString('WAVE');
    writeString('fmt ');
    view.setUint32(offset, 16, true);
    offset += 4;
    view.setUint16(offset, 1, true); // PCM format
    offset += 2;
    view.setUint16(offset, 1, true); // mono channel
    offset += 2;
    view.setUint32(offset, sampleRate, true);
    offset += 4;
    view.setUint32(offset, sampleRate * blockAlign, true);
    offset += 4;
    view.setUint16(offset, blockAlign, true);
    offset += 2;
    view.setUint16(offset, bytesPerSample * 8, true);
    offset += 2;
    writeString('data');
    view.setUint32(offset, samples.length * bytesPerSample, true);
    offset += 4;

    for (let i = 0; i < samples.length; i += 1) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        offset += 2;
    }

    return buffer;
};

const DemoPage = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [conversation, setConversation] = useState<ConversationEntry[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [statusMessage, setStatusMessage] = useState('Hold the button to record. Release to send.');

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const streamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const playbackUrlRef = useRef<string | null>(null);

    const instructions = useMemo(
        () => [
            'Allow microphone access when prompted.',
            'Press and hold the button while you speak.',
            'Release the button to send audio to Qwen.',
            'The response audio plays automatically.',
        ],
        [],
    );

    const resetStream = () => {
        mediaRecorderRef.current = null;
        chunksRef.current = [];
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }
    };

    const cleanup = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
        resetStream();
        if (audioContextRef.current) {
            void audioContextRef.current.close();
            audioContextRef.current = null;
        }
        if (playbackUrlRef.current) {
            URL.revokeObjectURL(playbackUrlRef.current);
            playbackUrlRef.current = null;
        }
    }, []);

    useEffect(() => () => cleanup(), [cleanup]);

    const ensureAudioContext = () => {
        if (!audioContextRef.current) {
            audioContextRef.current = getAudioContext();
        }
        return audioContextRef.current;
    };

    const convertToWavBlob = useCallback(async (blob: Blob): Promise<Blob> => {
        const ctx = ensureAudioContext();
        const arrayBuffer = await blob.arrayBuffer();
        const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));

        const { length, numberOfChannels, sampleRate } = decoded;
        const mono = new Float32Array(length);
        for (let channel = 0; channel < numberOfChannels; channel += 1) {
            const channelData = decoded.getChannelData(channel);
            for (let i = 0; i < length; i += 1) {
                mono[i] += channelData[i] / numberOfChannels;
            }
        }

        const wavBuffer = encodeWav(mono, sampleRate);
        return new Blob([wavBuffer], { type: 'audio/wav' });
    }, []);

    const playWavBlob = (blob: Blob) => {
        if (playbackUrlRef.current) {
            URL.revokeObjectURL(playbackUrlRef.current);
        }
        const url = URL.createObjectURL(blob);
        playbackUrlRef.current = url;
        const audio = new Audio(url);
        void audio.play();
    };

    const sendToBackend = useCallback(async (wavBlob: Blob) => {
        const form = new FormData();
        form.append('audio', wavBlob, 'recording.wav');
        if (sessionId) form.append('session_id', sessionId);

        const response = await fetch(`${API_BASE}/api/v1/audio`, { method: 'POST', body: form });

        if (!response.ok) {
            // Attempt to parse JSON error, else raw text
            let msg: string;
            try {
                const txt = await response.text();
                msg = txt || 'Backend request failed';
            } catch {
                msg = 'Backend request failed';
            }
            throw new Error(msg);
        }

        const contentType = response.headers.get('Content-Type') || '';
        if (!contentType.startsWith('audio/wav')) {
            throw new Error(`Unexpected response Content-Type: ${contentType}`);
        }

        const newSessionId = response.headers.get('X-Session-Id');
        if (!sessionId && newSessionId) setSessionId(newSessionId);
        const transcript = response.headers.get('X-Transcript') || '(no transcript)';

        // Add conversation entries
        setConversation((prev) => [
            ...prev,
            { role: 'user', text: '🎤 Sent an audio message' },
            { role: 'assistant', text: transcript },
        ]);

        const audioBlob = await response.blob();
        playWavBlob(audioBlob);
    }, [sessionId]);

    const processRecording = useCallback(async () => {
        const combined = new Blob(chunksRef.current, { type: 'audio/webm' });
        resetStream();
        if (combined.size === 0) {
            throw new Error('Captured audio was empty; please try again.');
        }
        const wavBlob = await convertToWavBlob(combined);
        await sendToBackend(wavBlob);
    }, [convertToWavBlob, sendToBackend]);

    const stopRecording = useCallback(() => {
        if (!mediaRecorderRef.current) {
            return;
        }
        if (mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
    }, []);

    const startRecording = useCallback(async () => {
        if (isProcessing) {
            return;
        }
        setError(null);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            const options: MediaRecorderOptions | undefined = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? { mimeType: 'audio/webm;codecs=opus' }
                : undefined;
            const recorder = new MediaRecorder(stream, options);
            chunksRef.current = [];

            recorder.ondataavailable = (event: BlobEvent) => {
                if (event.data.size > 0) {
                    chunksRef.current.push(event.data);
                }
            };

            recorder.onstop = async () => {
                setIsRecording(false);
                setIsProcessing(true);
                setStatusMessage('Sending audio to the model...');
                try {
                    await processRecording();
                    setStatusMessage('Hold the button to record. Release to send.');
                } catch (sendError) {
                    const message = sendError instanceof Error ? sendError.message : 'Failed to process audio.';
                    setError(message);
                    setStatusMessage('Hold the button to record. Release to send.');
                } finally {
                    setIsProcessing(false);
                }
            };

            recorder.start();
            mediaRecorderRef.current = recorder;
            setIsRecording(true);
            setStatusMessage('Recording...');
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Could not access the microphone.';
            setError(message);
            setIsRecording(false);
        }
    }, [isProcessing, processRecording]);

    const buttonLabel = isRecording ? 'Release to send' : isProcessing ? 'Processing audio…' : 'Hold to talk';

    return (
        <Stack spacing={4}>
            <Stack spacing={1}>
                <Typography variant="h4" fontWeight={700} component="h1">
                    Push-to-talk demo
                </Typography>
                <Typography variant="body1" color="text.secondary" maxWidth={620}>
                    Hold the microphone button to record audio. When you let go, the recording is sent to the backend
                    running Qwen2.5-Omni, which returns both text and synthesized speech that plays instantly.
                </Typography>
            </Stack>

            {error ? <Alert severity="error">{error}</Alert> : null}

            <Card>
                <CardContent>
                    <Stack spacing={3} alignItems="center">
                        <Typography variant="subtitle1" color="text.secondary">
                            {statusMessage}
                        </Typography>
                        <Button
                            variant="contained"
                            color={isRecording ? 'error' : 'primary'}
                            size="large"
                            disabled={isProcessing}
                            onMouseDown={startRecording}
                            onMouseUp={stopRecording}
                            onMouseLeave={isRecording ? stopRecording : undefined}
                            onTouchStart={startRecording}
                            onTouchEnd={stopRecording}
                            sx={{ px: 6, py: 2, textTransform: 'none', fontSize: '1.1rem', fontWeight: 600 }}
                        >
                            {buttonLabel}
                        </Button>
                        <Stack direction="row" spacing={1} flexWrap="wrap" justifyContent="center">
                            {instructions.map((tip) => (
                                <Chip key={tip} label={tip} color="default" />
                            ))}
                        </Stack>
                    </Stack>
                </CardContent>
            </Card>

            <Card>
                <CardContent>
                    <Stack spacing={2}>
                        <Typography variant="h6" component="h2">
                            Conversation
                        </Typography>
                        <Divider />
                        {conversation.length === 0 ? (
                            <Typography variant="body2" color="text.secondary">
                                Your conversation will appear here after you send your first message.
                            </Typography>
                        ) : (
                            <Stack spacing={1.5}>
                                {conversation.map((entry, idx) => (
                                    <Box
                                        key={`${entry.role}-${idx}`}
                                        sx={{
                                            bgcolor: entry.role === 'assistant' ? 'primary.main' : 'grey.900',
                                            color: 'common.white',
                                            px: 2,
                                            py: 1.5,
                                            borderRadius: 2,
                                        }}
                                    >
                                        <Typography variant="caption" sx={{ opacity: 0.8 }}>
                                            {entry.role === 'assistant' ? 'Qwen' : 'You'}
                                        </Typography>
                                        <Typography variant="body2">{entry.text}</Typography>
                                    </Box>
                                ))}
                            </Stack>
                        )}
                    </Stack>
                </CardContent>
            </Card>
        </Stack>
    );
};

export default DemoPage;
