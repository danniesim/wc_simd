'use client';

import AppBar from '@mui/material/AppBar';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Link from 'next/link';

const NAV_LINKS = [
    { label: 'Home', href: '/' },
    { label: 'Demo', href: '/demo' },
    { label: 'Docs', href: '/docs' },
];

const Header = () => {
    return (
        <AppBar position="static" color="primary" elevation={0}>
            <Toolbar sx={{ gap: 3 }}>
                <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 600 }}>
                    ImageTalk
                </Typography>
                <Stack direction="row" spacing={1.5} component="nav">
                    {NAV_LINKS.map((link) => (
                        <Button
                            key={link.href}
                            component={Link}
                            href={link.href}
                            color="inherit"
                            sx={{ textTransform: 'none', fontWeight: 500 }}
                        >
                            {link.label}
                        </Button>
                    ))}
                </Stack>
            </Toolbar>
        </AppBar>
    );
};

export default Header;
