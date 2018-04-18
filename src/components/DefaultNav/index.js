import React from "react";
import Link from "gatsby-link";
import styled from "styled-components";

const NavBar = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 40px;
  border: 1px black solid;
  > h1 {
    font-size: 20px;
    margin: 0;
    text-decoration: none;
  }
`;

const Nav = styled.div`
  display: flex;
`;

const SLink = styled(Link)`
  font-family: open sans;
  text-decoration: none;
  color: black;
  font-size: 16px;
  margin-left: 10px;
  &:hover {
    color: #28aa55;
  }
`;

export default ({ data }) => (
  <NavBar>
    <h1><SLink to={"/"}>Josh Meets Computer</SLink></h1>
    <Nav>
      <SLink to={"/projects/"}>Projects</SLink>
      <SLink to={"/notes/"}>Notes</SLink>
      <SLink to={"/letters/"}>Letters</SLink>
      <SLink to={"/about/"}>About</SLink>
    </Nav>
  </NavBar>
);
