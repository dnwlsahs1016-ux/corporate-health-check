import { Route, Routes } from "react-router-dom";
import Header from "./components/Header";
import CompanyDetail from "./pages/CompanyDetail";
import Home from "./pages/Home";

function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/companies/:corpName" element={<CompanyDetail />} />
      </Routes>
    </>
  );
}

export default App;
