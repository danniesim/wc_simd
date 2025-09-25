'use client';

import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Divider from '@mui/material/Divider';
import Typography from '@mui/material/Typography';
import type { ReactNode } from 'react';

import Header from './Header';

type AppShellProps = {
    children: ReactNode;
};

const AppShell = ({ children }: AppShellProps) => {
    return (
        <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', bgcolor: 'background.default' }}>
            <Header />
            <Container component="main" sx={{ flexGrow: 1, py: 6 }}>{children}</Container>
            <Divider />
            <Box component="footer" sx={{ py: 3, textAlign: 'center', bgcolor: 'background.paper' }}>
                <Typography variant="body2" color="text.secondary">
                    © {new Date().getFullYear()} ImageTalk. All rights reserved.
                </Typography>
            </Box>
        </Box>
    );
};

export default AppShell;
