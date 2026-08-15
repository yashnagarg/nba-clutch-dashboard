//purpose of this file is to handle all the api calls to the backend server, api service layer, ensures consistency and reusability of api calls across the application

const BASE_URL= import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function get(path)
{
    const response= await fetch(`${BASE_URL}${path}`);
    if(!response.ok)
    {
        throw new Error(`API error! status: ${response.status} on path: ${path}`);
    }
    return response.json();
}

//function to handle all the get requests
export const api={

}