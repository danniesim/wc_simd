import React from 'react';
import './globals.css';

const Layout = ({ children }) => {
    return (
        <div>
            <header>
                {/* You can include the Header component here */}
            </header>
            <main>{children}</main>
            <footer>
                {/* You can include footer content here */}
            </footer>
        </div>
    );
};

export default Layout;