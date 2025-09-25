'use client';

import createCache from '@emotion/cache';
import { CacheProvider } from '@emotion/react';
import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { useServerInsertedHTML } from 'next/navigation';
import { useState, type ReactNode } from 'react';

const theme = createTheme({
    palette: {
        mode: 'light',
        primary: {
            main: '#1a73e8',
        },
        secondary: {
            main: '#ff6d00',
        },
        background: {
            default: '#f5f7fb',
            paper: '#ffffff',
        },
    },
    typography: {
        fontFamily: ['"Roboto"', '"Helvetica"', '"Arial"', 'sans-serif'].join(', '),
    },
});

type ThemeRegistryProps = {
    children: ReactNode;
};

export function ThemeRegistry({ children }: ThemeRegistryProps) {
    const [cache] = useState(() => {
        const emotionCache = createCache({ key: 'mui', prepend: true });
        emotionCache.compat = true;
        return emotionCache;
    });

    useServerInsertedHTML(() => (
        <style
            data-emotion={`${cache.key} ${Object.keys(cache.inserted).join(' ')}`}
            dangerouslySetInnerHTML={{ __html: Object.values(cache.inserted).join('') }}
        />
    ));

    return (
        <CacheProvider value={cache}>
            <ThemeProvider theme={theme}>
                <CssBaseline />
                {children}
            </ThemeProvider>
        </CacheProvider>
    );
}

export default theme;
