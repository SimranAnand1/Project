const Footer = () => {
    return (
        <footer className="fixed-bottom text-center bg-light">
            <span>Made by </span>
            <a href={process.env.REACT_APP_AUTHOR_LINKEDIN_URL} target="_blank" rel="noreferrer">
                Vladislav Moroshan
            </a>

            <span> <i className="fa fa-sharp fa-solid fa-copyright"/> 2023</span>

        </footer>
    );
};

export default Footer;