import React from "react";
import Link from "gatsby-link";
import styled from "styled-components";

const Wrapper = styled.div`
  border-bottom: 1px solid grey;
  background-color: #28aa55;
`;

const NavBar = styled.div`
  max-width: 960px;
  flex-wrap: wrap;
  margin: auto;
  padding: 0px 10px 0 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  > h1 {
    padding: 0;
    margin: 0;
    line-height: 100%;
  }
`;

const Nav = styled.div`
  display: flex;
  padding: 8px 0 8px 0;
`;

const SLink = styled(Link)`
  font-family: open sans;
  text-decoration: none;
  color: white;
  font-size: 16px;
  margin: 0 0 0 10px;
  &:hover {
    color: #e4e4e4;
  }
`;

const TLink = styled(Link)`
  font-family: roboto;
  text-decoration: none;
  color: white;
  font-size: 20px;
  padding: 0;
  margin: 0
  &:hover {
    color: #e4e4e4;
  }
`;

export default ({ data }) => (
  <Wrapper>
    <NavBar>
      <h1>
        <TLink to={"/"}>Josh Meets Computer</TLink>
      </h1>
      <Nav>
        <SLink to={"/projects/"} style={{ marginLeft: 0 }}>Projects</SLink>
        <SLink to={"/notes/"}>Notes</SLink>
        <SLink to={"/letters/"}>Letters</SLink>
        <SLink to={"/about/"}>About</SLink>
      </Nav>
    </NavBar>
  </Wrapper>
);
