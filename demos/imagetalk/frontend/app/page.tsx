'use client';

import CollectionsIcon from '@mui/icons-material/Collections';
import ShareIcon from '@mui/icons-material/Share';
import SummarizeIcon from '@mui/icons-material/Summarize';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Link from 'next/link';
import type { ElementType } from 'react';

type Feature = {
    title: string;
    description: string;
    Icon: ElementType;
};

const FEATURES: Feature[] = [
    {
        title: 'Chat with Images',
        description: 'Upload an image and start a natural conversation powered by multi-modal AI.',
        Icon: CollectionsIcon,
    },
    {
        title: 'Smart Summaries',
        description: 'Generate concise descriptions, tags, and captions tailored to your workflow.',
        Icon: SummarizeIcon,
    },
    {
        title: 'Shareable Insights',
        description: 'Export conversations and highlights to collaborate with your team instantly.',
        Icon: ShareIcon,
    },
];

const HomePage = () => {
    return (
        <Stack spacing={6}>
            <Stack spacing={2} alignItems="flex-start">
                <Typography variant="h3" component="h1" fontWeight={700}>
                    Welcome to ImageTalk
                </Typography>
                <Typography variant="body1" color="text.secondary" maxWidth={600}>
                    ImageTalk blends visual understanding with conversational AI so you can explore images,
                    ask questions, and capture insights in seconds.
                </Typography>
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                    <Button variant="contained" color="primary" size="large" component={Link} href="/demo">
                        Try the demo
                    </Button>
                    <Button variant="outlined" color="secondary" size="large" component={Link} href="/docs">
                        View the docs
                    </Button>
                </Stack>
            </Stack>

            <Grid container spacing={3}>
                {FEATURES.map(({ title, description, Icon }) => (
                    <Grid item xs={12} md={4} key={title}>
                        <Card elevation={1} sx={{ height: '100%' }}>
                            <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                                <Icon color="primary" fontSize="large" />
                                <Typography variant="h6" component="h3" fontWeight={600}>
                                    {title}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    {description}
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>
                ))}
            </Grid>
        </Stack>
    );
};

export default HomePage;
