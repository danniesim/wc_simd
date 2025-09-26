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

type AudioHistoryEntry = {
    id: string;
    url: string;
    transcript: string;
    timestamp: number;
};

type ImageSearchFollowUp = {
    id: string;
    transcript: string;
    audioUrl: string;
    timestamp: number;
};

type ImageSearchResult = {
    id: string;
    transcript: string;
    imageIds: string[];
    timestamp: number;
    followUps: ImageSearchFollowUp[];
};

type RecordingMode = 'audio' | 'images' | 'imageQuery';

type RecordingContext = {
    mode: RecordingMode;
    imageUrls?: string[];
    resultId?: string;
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

const AUDIO_DEFAULT_STATUS = 'Hold the button to record. Release to send.';
const IMAGE_DEFAULT_STATUS = 'Hold the button to capture audio for image search.';

const DemoPage = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [audioProcessing, setAudioProcessing] = useState(false);
    const [imageProcessing, setImageProcessing] = useState(false);
    const [audioHistory, setAudioHistory] = useState<AudioHistoryEntry[]>([]);
    const [imageResults, setImageResults] = useState<ImageSearchResult[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [audioStatus, setAudioStatus] = useState(AUDIO_DEFAULT_STATUS);
    const [imageStatus, setImageStatus] = useState(IMAGE_DEFAULT_STATUS);
    const [imageQueryStatus, setImageQueryStatus] = useState<Record<string, string>>({});
    const [imageQueryProcessingId, setImageQueryProcessingId] = useState<string | null>(null);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const streamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const audioUrlsRef = useRef<string[]>([]);
    const recordingContextRef = useRef<RecordingContext | null>(null);

    const audioTips = useMemo(
        () => [
            'Allow microphone access when prompted.',
            'Press and hold the button while you speak.',
            'Release the button to send audio to Qwen.',
            'The response audio plays automatically.',
        ],
        [],
    );

    const imageTips = useMemo(
        () => [
            'Press and hold to capture a query for image search.',
            'We transcribe your speech before embedding.',
            'Returns the top 4 matching image IDs from the gallery.',
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
        recordingContextRef.current = null;
    }, []);

    useEffect(
        () => () => {
            cleanup();
            audioUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
            audioUrlsRef.current = [];
        },
        [cleanup],
    );

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

    const registerAudioBlob = useCallback((blob: Blob): string => {
        const url = URL.createObjectURL(blob);
        audioUrlsRef.current.push(url);
        return url;
    }, []);

    const playAudioUrl = useCallback((url: string) => {
        const audio = new Audio(url);
        void audio.play();
    }, []);

    const sendToAudioBackend = useCallback(async (wavBlob: Blob) => {
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

        const audioBlob = await response.blob();
        const url = registerAudioBlob(audioBlob);
        const entry: AudioHistoryEntry = {
            id: `resp-${Date.now()}`,
            url,
            transcript,
            timestamp: Date.now(),
        };
        setAudioHistory((prev) => [...prev, entry]);
        playAudioUrl(url);
    }, [playAudioUrl, registerAudioBlob, sessionId]);

    const sendToImageSearch = useCallback(async (wavBlob: Blob) => {
        const form = new FormData();
        form.append('audio', wavBlob, 'recording.wav');

        const response = await fetch(`${API_BASE}/api/v1/images/from-audio`, { method: 'POST', body: form });
        if (!response.ok) {
            let msg: string;
            try {
                const txt = await response.text();
                msg = txt || 'Image search request failed';
            } catch {
                msg = 'Image search request failed';
            }
            throw new Error(msg);
        }

        const payload = await response.json();
        const transcript: string = typeof payload?.transcript === 'string' ? payload.transcript : '';
        const rawImageIds = payload?.image_ids ?? payload?.imageIds;
        const imageIds: string[] = Array.isArray(rawImageIds)
            ? rawImageIds.map((id: unknown) => (id == null ? '' : String(id))).filter((id: string) => id.trim() !== '')
            : [];

        const entry: ImageSearchResult = {
            id: `img-${Date.now()}`,
            transcript,
            imageIds: imageIds.slice(0, 4),
            timestamp: Date.now(),
            followUps: [],
        };
        setImageResults((prev) => [entry, ...prev]);
        setImageQueryStatus((prev) => ({
            ...prev,
            [entry.id]: 'Hold the button to ask about these images.',
        }));
    }, []);

    const sendToAudioImage = useCallback(
        async (wavBlob: Blob, imageUrls: string[], resultId?: string) => {
            if (imageUrls.length === 0) {
                throw new Error('No images available for this query.');
            }
            const form = new FormData();
            form.append('audio', wavBlob, 'recording.wav');
            imageUrls.forEach((url) => {
                if (url) {
                    form.append('images', url);
                }
            });

            const response = await fetch(`${API_BASE}/api/v1/audio-image`, {
                method: 'POST',
                body: form,
            });

            if (!response.ok) {
                let msg: string;
                try {
                    const txt = await response.text();
                    msg = txt || 'Audio-image request failed';
                } catch {
                    msg = 'Audio-image request failed';
                }
                throw new Error(msg);
            }

            const contentType = response.headers.get('Content-Type') || '';
            if (!contentType.startsWith('audio/wav')) {
                let msg: string;
                try {
                    msg = await response.text();
                } catch {
                    msg = 'Unexpected response type from audio-image endpoint';
                }
                throw new Error(msg || 'Unexpected response type from audio-image endpoint');
            }

            const transcript = response.headers.get('X-Transcript') || '(no transcript)';
            const audioBlob = await response.blob();
            const url = registerAudioBlob(audioBlob);
            const followUp: ImageSearchFollowUp = {
                id: `img-follow-${Date.now()}`,
                transcript,
                audioUrl: url,
                timestamp: Date.now(),
            };

            if (resultId) {
                setImageResults((prev) =>
                    prev.map((result) =>
                        result.id === resultId
                            ? {
                                  ...result,
                                  followUps: [...result.followUps, followUp],
                              }
                            : result,
                    ),
                );
                setImageQueryStatus((prev) => ({
                    ...prev,
                    [resultId]: 'Hold the button to ask about these images.',
                }));
            }

            playAudioUrl(url);
        },
        [playAudioUrl, registerAudioBlob],
    );

    const finishRecording = useCallback(
        async (context: RecordingContext) => {
            const combined = new Blob(chunksRef.current, { type: 'audio/webm' });
            resetStream();
            if (combined.size === 0) {
                throw new Error('Captured audio was empty; please try again.');
            }
            const wavBlob = await convertToWavBlob(combined);
            if (context.mode === 'audio') {
                await sendToAudioBackend(wavBlob);
            } else if (context.mode === 'images') {
                await sendToImageSearch(wavBlob);
            } else {
                await sendToAudioImage(
                    wavBlob,
                    context.imageUrls ?? [],
                    context.resultId,
                );
            }
        },
        [convertToWavBlob, sendToAudioBackend, sendToAudioImage, sendToImageSearch],
    );

    const stopRecording = useCallback(() => {
        if (!mediaRecorderRef.current) {
            return;
        }
        if (mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
    }, []);

    const startRecording = useCallback(
        async (mode: RecordingMode, extra?: { imageUrls?: string[]; resultId?: string }) => {
            if (isRecording || audioProcessing || imageProcessing || imageQueryProcessingId) {
                return;
            }
            let contextExtra = extra;
            if (mode === 'imageQuery') {
                const urls = (extra?.imageUrls ?? []).filter((url) => url && url.trim() !== '');
                if (urls.length === 0) {
                    setError('No images available for this query.');
                    return;
                }
                contextExtra = { ...extra, imageUrls: urls };
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
                    const context = recordingContextRef.current;
                    setIsRecording(false);
                    if (!context) {
                        recordingContextRef.current = null;
                        return;
                    }

                    if (context.mode === 'audio') {
                        setAudioProcessing(true);
                        setAudioStatus('Sending audio to the model...');
                    } else if (context.mode === 'images') {
                        setImageProcessing(true);
                        setImageStatus('Searching for similar images...');
                    } else if (context.resultId) {
                        setImageQueryProcessingId(context.resultId);
                        setImageQueryStatus((prev) => ({
                            ...prev,
                            [context.resultId as string]: 'Sending audio with selected images to the model...',
                        }));
                    }

                    try {
                        await finishRecording(context);
                        if (context.mode === 'audio') {
                            setAudioStatus(AUDIO_DEFAULT_STATUS);
                        } else if (context.mode === 'images') {
                            setImageStatus(IMAGE_DEFAULT_STATUS);
                        }
                    } catch (sendError) {
                        const message = sendError instanceof Error ? sendError.message : 'Failed to process audio.';
                        setError(message);
                        if (context.mode === 'audio') {
                            setAudioStatus(AUDIO_DEFAULT_STATUS);
                        } else if (context.mode === 'images') {
                            setImageStatus(IMAGE_DEFAULT_STATUS);
                        } else if (context.resultId) {
                            setImageQueryStatus((prev) => ({
                                ...prev,
                                [context.resultId as string]: message,
                            }));
                        }
                    } finally {
                        if (context.mode === 'audio') {
                            setAudioProcessing(false);
                        } else if (context.mode === 'images') {
                            setImageProcessing(false);
                        } else if (context.resultId) {
                            setImageQueryProcessingId(null);
                            setImageQueryStatus((prev) => {
                                const current = prev[context.resultId as string];
                                const shouldReset =
                                    !current ||
                                    current === 'Recording... Release to send.' ||
                                    current === 'Sending audio with selected images to the model...';
                                return {
                                    ...prev,
                                    [context.resultId as string]: shouldReset
                                        ? 'Hold the button to ask about these images.'
                                        : current,
                                };
                            });
                        }
                        recordingContextRef.current = null;
                    }
                };

                recorder.start();
                mediaRecorderRef.current = recorder;
                recordingContextRef.current = {
                    mode,
                    imageUrls: contextExtra?.imageUrls,
                    resultId: contextExtra?.resultId,
                };
                setIsRecording(true);
                if (mode === 'audio') {
                    setAudioStatus('Recording...');
                } else if (mode === 'images') {
                    setImageStatus('Recording audio for image search...');
                } else if (contextExtra?.resultId) {
                    setImageQueryStatus((prev) => ({
                        ...prev,
                        [contextExtra.resultId as string]: 'Recording... Release to send.',
                    }));
                }
            } catch (err) {
                const message = err instanceof Error ? err.message : 'Could not access the microphone.';
                setError(message);
                setIsRecording(false);
                recordingContextRef.current = null;
                if (mode === 'audio') {
                    setAudioStatus(AUDIO_DEFAULT_STATUS);
                } else if (mode === 'images') {
                    setImageStatus(IMAGE_DEFAULT_STATUS);
                } else if (contextExtra?.resultId) {
                    setImageQueryStatus((prev) => ({
                        ...prev,
                        [contextExtra.resultId as string]: message,
                    }));
                }
            }
        },
        [audioProcessing, finishRecording, imageProcessing, imageQueryProcessingId, isRecording],
    );

    const audioButtonLabel =
        isRecording && recordingContextRef.current?.mode === 'audio'
            ? 'Release to send'
            : audioProcessing
                ? 'Processing audio…'
                : 'Hold to talk';

    const imageButtonLabel =
        isRecording && recordingContextRef.current?.mode === 'images'
            ? 'Release to search'
            : imageProcessing
                ? 'Searching images…'
                : 'Hold to image search';

    const audioButtonDisabled =
        audioProcessing ||
        imageProcessing ||
        imageQueryProcessingId !== null ||
        (isRecording && recordingContextRef.current?.mode !== 'audio');
    const imageButtonDisabled =
        audioProcessing ||
        imageProcessing ||
        imageQueryProcessingId !== null ||
        (isRecording && recordingContextRef.current?.mode !== 'images');

    const handleAudioStart = useCallback(() => startRecording('audio'), [startRecording]);
    const handleAudioStop = useCallback(() => {
        if (recordingContextRef.current?.mode === 'audio') {
            stopRecording();
        }
    }, [stopRecording]);

    const handleImageStart = useCallback(() => startRecording('images'), [startRecording]);
    const handleImageStop = useCallback(() => {
        if (recordingContextRef.current?.mode === 'images') {
            stopRecording();
        }
    }, [stopRecording]);

    const handleImageQueryStart = useCallback(
        (resultId: string, imageUrls: string[]) => startRecording('imageQuery', { resultId, imageUrls }),
        [startRecording],
    );

    const handleImageQueryStop = useCallback(
        (resultId: string) => {
            const context = recordingContextRef.current;
            if (context?.mode === 'imageQuery' && context.resultId === resultId) {
                stopRecording();
            }
        },
        [stopRecording],
    );

    const handlePlayHistory = useCallback(
        (url: string) => {
            playAudioUrl(url);
        },
        [playAudioUrl],
    );

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
                            {audioStatus}
                        </Typography>
                        <Button
                            variant="contained"
                            color={
                                isRecording && recordingContextRef.current?.mode === 'audio'
                                    ? 'error'
                                    : 'primary'
                            }
                            size="large"
                            disabled={audioButtonDisabled}
                            onMouseDown={handleAudioStart}
                            onMouseUp={handleAudioStop}
                            onMouseLeave={
                                isRecording && recordingContextRef.current?.mode === 'audio'
                                    ? handleAudioStop
                                    : undefined
                            }
                            onTouchStart={handleAudioStart}
                            onTouchEnd={handleAudioStop}
                            sx={{ px: 6, py: 2, textTransform: 'none', fontSize: '1.1rem', fontWeight: 600 }}
                        >
                            {audioButtonLabel}
                        </Button>
                        <Stack direction="row" spacing={1} flexWrap="wrap" justifyContent="center">
                            {audioTips.map((tip) => (
                                <Chip key={tip} label={tip} color="default" />
                            ))}
                        </Stack>
                    </Stack>
                </CardContent>
            </Card>

            <Card>
                <CardContent>
                    <Stack spacing={3} alignItems="center">
                        <Typography variant="subtitle1" color="text.secondary">
                            {imageStatus}
                        </Typography>
                        <Button
                            variant="contained"
                            color={
                                isRecording && recordingContextRef.current?.mode === 'images'
                                    ? 'error'
                                    : 'secondary'
                            }
                            size="large"
                            disabled={imageButtonDisabled}
                            onMouseDown={handleImageStart}
                            onMouseUp={handleImageStop}
                            onMouseLeave={
                                isRecording && recordingContextRef.current?.mode === 'images'
                                    ? handleImageStop
                                    : undefined
                            }
                            onTouchStart={handleImageStart}
                            onTouchEnd={handleImageStop}
                            sx={{ px: 6, py: 2, textTransform: 'none', fontSize: '1.1rem', fontWeight: 600 }}
                        >
                            {imageButtonLabel}
                        </Button>
                        <Stack direction="row" spacing={1} flexWrap="wrap" justifyContent="center">
                            {imageTips.map((tip) => (
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
                            Assistant Audio History
                        </Typography>
                        <Divider />
                        {audioHistory.length === 0 ? (
                            <Typography variant="body2" color="text.secondary">
                                Assistant responses will appear here, ready for replay.
                            </Typography>
                        ) : (
                            <Stack spacing={1.5}>
                                {audioHistory.map((entry) => (
                                    <Box
                                        key={entry.id}
                                        sx={{
                                            bgcolor: 'grey.900',
                                            color: 'common.white',
                                            px: 2,
                                            py: 1.5,
                                            borderRadius: 2,
                                            display: 'flex',
                                            flexDirection: 'column',
                                            gap: 1,
                                        }}
                                    >
                                        <Stack direction="row" spacing={1.5} alignItems="center">
                                            <Button
                                                variant="outlined"
                                                color="inherit"
                                                size="small"
                                                onClick={() => handlePlayHistory(entry.url)}
                                                sx={{ textTransform: 'none' }}
                                            >
                                                ▶ Play
                                            </Button>
                                            <Typography variant="caption" sx={{ opacity: 0.8 }}>
                                                {new Date(entry.timestamp).toLocaleTimeString()}
                                            </Typography>
                                        </Stack>
                                        <Typography variant="body2">{entry.transcript || '(no transcript)'}</Typography>
                                    </Box>
                                ))}
                            </Stack>
                        )}
                    </Stack>
                </CardContent>
            </Card>

            <Card>
                <CardContent>
                    <Stack spacing={2}>
                        <Typography variant="h6" component="h2">
                            Image Search Results
                        </Typography>
                        <Divider />
                        {imageResults.length === 0 ? (
                            <Typography variant="body2" color="text.secondary">
                                Use the image search push-to-talk button to fetch matching image IDs.
                            </Typography>
                        ) : (
                            <Stack spacing={1.5}>
                                {imageResults.map((result) => {
                                    const queryStatusMessage =
                                        imageQueryStatus[result.id] ?? 'Hold the button to ask about these images.';
                                    const recordingContext = recordingContextRef.current;
                                    const isGridRecording =
                                        isRecording &&
                                        recordingContext?.mode === 'imageQuery' &&
                                        recordingContext?.resultId === result.id;
                                    const isGridProcessing = imageQueryProcessingId === result.id;
                                    const gridButtonDisabled =
                                        audioProcessing ||
                                        imageProcessing ||
                                        isGridProcessing ||
                                        imageQueryProcessingId !== null ||
                                        (isRecording && !isGridRecording);
                                    const gridButtonLabel = isGridRecording
                                        ? 'Release to query'
                                        : isGridProcessing
                                            ? 'Processing query…'
                                            : imageQueryProcessingId !== null
                                                ? 'Please wait…'
                                                : 'Hold to ask about these images';

                                    return (
                                        <Box
                                            key={result.id}
                                            sx={{
                                                bgcolor: 'grey.100',
                                                color: 'grey.900',
                                                px: 2,
                                                py: 1.5,
                                                borderRadius: 2,
                                                border: '1px solid',
                                                borderColor: 'grey.300',
                                                display: 'flex',
                                                flexDirection: 'column',
                                                gap: 1.25,
                                            }}
                                        >
                                            <Typography variant="caption" color="text.secondary">
                                                {new Date(result.timestamp).toLocaleTimeString()}
                                            </Typography>
                                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                                Transcript: {result.transcript || '(empty)'}
                                            </Typography>
                                            {result.imageIds.length > 0 ? (
                                                <Box
                                                    sx={{
                                                        mt: 0.5,
                                                        display: 'grid',
                                                        gap: 1.5,
                                                        gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                                                    }}
                                                >
                                                    {result.imageIds.map((url) => (
                                                        <Box
                                                            key={url}
                                                            sx={{
                                                                position: 'relative',
                                                                borderRadius: 2,
                                                                overflow: 'hidden',
                                                                bgcolor: 'grey.200',
                                                                aspectRatio: '1 / 1',
                                                            }}
                                                        >
                                                            <Box
                                                                component="img"
                                                                src={url}
                                                                alt="Image search result"
                                                                sx={{
                                                                    width: '100%',
                                                                    height: '100%',
                                                                    objectFit: 'cover',
                                                                    display: 'block',
                                                                }}
                                                            />
                                                        </Box>
                                                    ))}
                                                </Box>
                                            ) : (
                                                <Typography variant="body2">
                                                    Top image IDs: None returned
                                                </Typography>
                                            )}

                                            <Stack spacing={1} alignItems="flex-start">
                                                <Button
                                                    variant="contained"
                                                    color={isGridRecording ? 'error' : 'primary'}
                                                    size="small"
                                                    disabled={gridButtonDisabled}
                                                    onMouseDown={() => handleImageQueryStart(result.id, result.imageIds)}
                                                    onMouseUp={() => handleImageQueryStop(result.id)}
                                                    onMouseLeave={isGridRecording ? () => handleImageQueryStop(result.id) : undefined}
                                                    onTouchStart={() => handleImageQueryStart(result.id, result.imageIds)}
                                                    onTouchEnd={() => handleImageQueryStop(result.id)}
                                                    sx={{ textTransform: 'none', fontWeight: 600 }}
                                                >
                                                    {gridButtonLabel}
                                                </Button>
                                                <Typography variant="body2" color="text.secondary">
                                                    {queryStatusMessage}
                                                </Typography>
                                            </Stack>

                                            {result.followUps.length > 0 ? (
                                                <Stack spacing={1.25} sx={{ mt: 0.5 }}>
                                                    {result.followUps.map((follow) => (
                                                        <Box
                                                            key={follow.id}
                                                            sx={{
                                                                display: 'flex',
                                                                alignItems: 'center',
                                                                gap: 1,
                                                                flexWrap: 'wrap',
                                                            }}
                                                        >
                                                            <Button
                                                                variant="outlined"
                                                                color="inherit"
                                                                size="small"
                                                                onClick={() => handlePlayHistory(follow.audioUrl)}
                                                                sx={{ textTransform: 'none' }}
                                                            >
                                                                ▶ Play
                                                            </Button>
                                                            <Typography variant="caption" color="text.secondary">
                                                                {new Date(follow.timestamp).toLocaleTimeString()}
                                                            </Typography>
                                                            <Typography variant="body2">{follow.transcript}</Typography>
                                                        </Box>
                                                    ))}
                                                </Stack>
                                            ) : null}
                                        </Box>
                                    );
                                })}
                            </Stack>
                        )}
                    </Stack>
                </CardContent>
            </Card>
        </Stack>
    );
};

export default DemoPage;
