export const industries = [
  "IT Services", "Software Development", "SaaS", "FinTech", "E-commerce", 
  "EdTech", "HealthTech", "Cloud Computing", "Cybersecurity", "Data Analytics", 
  "Artificial Intelligence", "Machine Learning", "Blockchain/Web3", "IoT", 
  "Robotics", "AR/VR", "Enterprise Software", "Digital Platforms", 
  "Startup Ecosystem", "Product-Based Companies"
];

export const companies = [
  // Kerala
  "UST", "IBS Software", "NeST Digital", "Experion", "Quest Global", "SurveySparrow",
  // India - Service
  "TCS", "Infosys", "Wipro", "HCLTech", "Cognizant",
  // India - Product/Startup
  "Flipkart", "Razorpay", "CRED", "Meesho", "Paytm", "Zoho", "Freshworks",
  // Global
  "Google", "Microsoft", "Amazon", "Apple", "Meta", "Adobe", "Netflix", "Oracle", "Salesforce", "NVIDIA", "OpenAI"
];

/**
 * Smart Mapping: Role-Specific Market Data
 */
export const marketData = {
  "Frontend Developer": {
    demand: "High",
    competition: "Medium",
    trend: "Rising",
    barrier: "Medium",
    salaries: { entry: "4-8 LPA", mid: "12-18 LPA", top: "30+ LPA" },
    industries: ["SaaS", "E-commerce", "AI Products", "Digital Platforms"],
    companies: ["Google", "Microsoft", "Adobe", "Flipkart", "Zoho", "Freshworks", "SurveySparrow"],
    insight: "High demand for modern frontend frameworks like React. UI/UX and performance optimization are key skills."
  },

  "Backend Developer": {
    demand: "High",
    competition: "Medium",
    trend: "Rising",
    barrier: "Medium",
    salaries: { entry: "5-10 LPA", mid: "15-25 LPA", top: "40+ LPA" },
    industries: ["Cloud Computing", "FinTech", "AI Platforms", "Enterprise Software"],
    companies: ["Amazon", "Uber", "Paytm", "Razorpay", "Netflix", "Oracle", "Salesforce", "NeST Digital"],
    insight: "Backend roles require strong API design, database management, and system scalability knowledge."
  },

  "Full Stack Developer": {
    demand: "Very High",
    competition: "Medium",
    trend: "Rising",
    barrier: "Medium",
    salaries: { entry: "5-9 LPA", mid: "15-22 LPA", top: "35+ LPA" },
    industries: ["Startups", "SaaS", "FinTech", "E-commerce"],
    companies: ["Flipkart", "Razorpay", "CRED", "Meesho", "Zoho", "Freshworks", "UST", "Quest Global"],
    insight: "Full stack developers are highly valued for handling both frontend and backend systems in startups and product companies."
  },

  "AI Engineer": {
    demand: "High",
    competition: "High",
    trend: "Rising",
    barrier: "High",
    salaries: { entry: "8-15 LPA", mid: "20-35 LPA", top: "50+ LPA" },
    industries: ["AI/ML", "Healthcare Tech", "Autonomous Systems", "Data Platforms"],
    companies: ["Google", "Microsoft", "Amazon", "OpenAI", "NVIDIA", "Meta", "Adobe", "NeST Digital"],
    insight: "AI roles require strong math, ML models, and real-world implementation experience."
  }
};
