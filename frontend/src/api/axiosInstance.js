import axios from 'axios'

const isDevEnv = !process.env.NODE_ENV || process.env.NODE_ENV === 'development';

const axiosInstance = axios.create({
    baseURL: isDevEnv ? "http://localhost:8000/" : "/",
    timeout: 60000
});


export default axiosInstance;